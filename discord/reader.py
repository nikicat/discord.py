# -*- coding: utf-8 -*-

"""
The MIT License (MIT)

Copyright (c) 2015-2019 Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import asyncio
import time
import wave
import socket
import audioop
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

from nacl.exceptions import CryptoError

from . import rtp
from .opus import Decoder
from .errors import DiscordException, ClientException
from .member import Member
from .rtp import RTPPacket

log = logging.getLogger(__name__)

__all__ = [
    'AudioSink',
    'WaveSink',
    'PCMVolumeTransformerFilter',
    'SinkExit'
]


class SinkExit(DiscordException):
    """A signal type exception (like ``GeneratorExit``) to raise in a Sink's write() method to stop it."""


@dataclass
class VoiceData:
    pcm: bytes
    user: Member
    packet: RTPPacket


class AudioSink:
    def __del__(self):
        self.cleanup()

    def write(self, data: VoiceData):
        raise NotImplementedError

    def speech_started(self, user: Member):
        raise NotImplementedError

    def speech_ended(self, user: Member):
        raise NotImplementedError

    def cleanup(self):
        pass


class WaveSink(AudioSink):
    def __init__(self, destination):
        self._file = wave.open(destination, 'wb')
        self._file.setnchannels(Decoder.CHANNELS)
        self._file.setsampwidth(Decoder.SAMPLE_SIZE // Decoder.CHANNELS)
        self._file.setframerate(Decoder.SAMPLING_RATE)

    def write(self, data):
        self._file.writeframes(data.data)

    def cleanup(self):
        self._file.close()


class PCMVolumeTransformerFilter(AudioSink):
    def __init__(self, destination, volume=1.0):
        if not isinstance(destination, AudioSink):
            raise TypeError('expected AudioSink not {0.__class__.__name__}.'.format(destination))

        if destination.wants_opus():
            raise ClientException('AudioSink must not request Opus encoding.')

        self.destination = destination
        self.volume = volume

    @property
    def volume(self):
        """Retrieves or sets the volume as a floating point percentage (e.g. 1.0 for 100%)."""
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = max(value, 0.0)

    def write(self, data):
        data = audioop.mul(data.data, 2, min(self._volume, 2.0))
        self.destination.write(data)


class AudioReader:
    def __init__(self, sink, client):
        self.sink = sink
        self.client = client
        self.decoders = defaultdict(partial(BufferedAudioDecoder, self))
        self.decrypt_rtp = getattr(self, '_decrypt_rtp_' + client._mode)
        self.decrypt_rtcp = getattr(self, '_decrypt_rtcp_' + client._mode)

    def start(self):
        self.loop = asyncio.create_task(self.run())

    def stop(self):
        self.loop.cancel()

    @property
    def connected(self):
        return self.client._connected

    def _decrypt_rtp_xsalsa20_poly1305(self, packet):
        nonce = bytearray(24)
        nonce[:12] = packet.header
        result = self.client.box.decrypt(bytes(packet.data), bytes(nonce))

        if packet.extended:
            offset = packet.update_ext_headers(result)
            result = result[offset:]

        return result

    def _decrypt_rtcp_xsalsa20_poly1305(self, data):
        nonce = bytearray(24)
        nonce[:8] = data[:8]
        result = self.client.box.decrypt(data[8:], bytes(nonce))

        return data[:8] + result

    def _decrypt_rtp_xsalsa20_poly1305_suffix(self, packet):
        nonce = packet.data[-24:]
        voice_data = packet.data[:-24]
        result = self.client.box.decrypt(bytes(voice_data), bytes(nonce))

        if packet.extended:
            offset = packet.update_ext_headers(result)
            result = result[offset:]

        return result

    def _decrypt_rtcp_xsalsa20_poly1305_suffix(self, data):
        nonce = data[-24:]
        header = data[:8]
        result = self.client.box.decrypt(data[8:-24], nonce)

        return header + result

    def _decrypt_rtp_xsalsa20_poly1305_lite(self, packet):
        nonce = bytearray(24)
        nonce[:4] = packet.data[-4:]
        voice_data = packet.data[:-4]
        result = self.client.box.decrypt(bytes(voice_data), bytes(nonce))

        if packet.extended:
            offset = packet.update_ext_headers(result)
            result = result[offset:]

        return result

    def _decrypt_rtcp_xsalsa20_poly1305_lite(self, data):
        nonce = bytearray(24)
        nonce[:4] = data[-4:]
        header = data[:8]
        result = self.client.box.decrypt(data[8:-4], bytes(nonce))

        return header + result

    def _reset_decoders(self, ssrc=None):
        log.debug(f"Reseting decoder(s) {ssrc}")
        if ssrc in self.decoders:
            del self.decoders[ssrc]
        else:
            self.decoders.clear()

    def _stop_decoders(self):
        log.debug("Stopping decoders")
        self.decoders.clear()

    def _ssrc_removed(self, ssrc):
        log.debug(f"Removed ssrc {ssrc}")
        if ssrc in self.decoders:
            del self.decoders[ssrc]

    def _get_user(self, ssrc):
        _, user_id = self.client._get_ssrc_mapping(ssrc=ssrc)
        # may need to change this for calls or something
        return self.client.guild.get_member(user_id)

    def feed(self, packet, pcm):
        try:
            user = self._get_user(packet.ssrc)
            self.sink.write(VoiceData(pcm, user, packet))
        except SinkExit:
            log.info("Shutting down reader thread %s", self)
            self.stop()
            self._stop_decoders()
        except Exception as exc:
            log.exception(f"Exception when writing to sink: {exc}")

    def speech_started(self, ssrc):
        self.sink.speech_started(self._get_user(ssrc))

    def speech_ended(self, ssrc):
        self.sink.speech_ended(self._get_user(ssrc))

    def _set_sink(self, sink):
        self.sink = sink

    async def _do_run(self):
        loop = asyncio.get_running_loop()
        while True:
            try:
                raw_data = await loop.sock_recv(self.client.socket, 10000)
            except socket.error as e:
                t0 = time.time()

                log.exception(f"Socket error in reader loop: {e}")
                await self.client._connecting.wait()

                if self.client.is_connected():
                    log.debug(f"Reconnected in {time.time()-t0:.4f}s")
                    continue
                else:
                    raise

            try:
                if not rtp.is_rtcp(raw_data):
                    packet = rtp.decode(raw_data)
                    packet.decrypted_data = self.decrypt_rtp(packet)
                else:
                    packet = rtp.decode(self.decrypt_rtcp(raw_data))
                    log.debug(f"Received rtcp: {packet}")
                    if not isinstance(packet, rtp.ReceiverReportPacket):
                        log.warning(f"Not a ReceiverReportPacket: {packet}")
                        # TODO: Fabricate and send SenderReports and see what happens
                    continue

            except CryptoError:
                log.exception("CryptoError decoding packet %s", raw_data)
                continue

            except Exception as exc:
                log.exception(f"Error unpacking packet: {exc}")

            else:
                if packet.ssrc not in self.client._ssrcs:
                    log.warning("Received packet for unknown ssrc %s", packet.ssrc)
                else:
                    self.decoders[packet.ssrc].feed(packet)

    async def run(self):
        try:
            await self._do_run()
        except Exception as exc:
            log.exception(f"Exception in _do_run(): {exc}")
            raise
        finally:
            self._stop_decoders()
            try:
                self.sink.cleanup()
            except Exception as exc:
                log.exception(f"Error during sink cleanup: {exc}")

    def is_listening(self):
        return not self._end.is_set()


class BufferedAudioDecoder:
    def __init__(self, reader: AudioReader):
        self.reader = reader
        self.decoder = Decoder()
        self.next_seq = 0
        self.silencer = None

    def feed(self, packet: RTPPacket):
        if packet.sequence == self.next_seq:
            self.next_seq += 1
            pcm = self.decoder.decode(packet.decrypted_data)
        elif packet.sequence > self.next_seq:
            self.next_seq = packet.sequence + 1
            pcm = self.decoder.decode(packet.decrypted_data, fec=True)
            log.debug(f"Received packet with a gap {packet.sequence - self.next_seq}, using FEC")
        elif packet.sequence < self.next_seq:
            log.debug(f"Received out-of-order packet {self.next_seq - packet.sequence}, skipping")
            return
        if self.silencer is None:
            self.reader.speech_started(packet.ssrc)
        else:
            self.silencer.cancel()
        self.silencer = asyncio.create_task(self.end_speech(packet.ssrc))
        self.reader.feed(packet, pcm)

    async def end_speech(self, ssrc):
        await asyncio.sleep(0.05)
        self.reader.speech_ended(ssrc)
        self.silencer = None
