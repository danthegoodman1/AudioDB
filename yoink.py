import fnmatch
import os
from hashlib import sha1
from itertools import groupby
from typing import List, Tuple, Dict

import numpy as np
from pydub import AudioSegment
from pydub.utils import audioop

import wavio

from hashlib import sha256
from operator import itemgetter
from typing import List, Tuple

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)

HASHES_MATCHED = 'hashes_matched_in_input'

# Hashes fingerprinted in the db.
FINGERPRINTED_HASHES = 'fingerprinted_hashes_in_db'
# Percentage regarding hashes matched vs hashes fingerprinted in the db.
FINGERPRINTED_CONFIDENCE = 'fingerprinted_confidence'

# Hashes generated from the input.
INPUT_HASHES = 'input_total_hashes'
# Percentage regarding hashes matched vs hashes from the input.
INPUT_CONFIDENCE = 'input_confidence'

TOTAL_TIME = 'total_time'
FINGERPRINT_TIME = 'fingerprint_time'
QUERY_TIME = 'query_time'
ALIGN_TIME = 'align_time'
OFFSET = 'offset'
OFFSET_SECS = 'offset_seconds'

# DATABASE CLASS INSTANCES:
DATABASES = {
    'mysql': ("dejavu.database_handler.mysql_database", "MySQLDatabase"),
    'postgres': ("dejavu.database_handler.postgres_database", "PostgreSQLDatabase")
}

# TABLE SONGS
SONGS_TABLENAME = "songs"

# SONGS FIELDS
FIELD_SONG_ID = 'song_id'
FIELD_SONGNAME = 'song_name'
FIELD_FINGERPRINTED = "fingerprinted"
FIELD_FILE_SHA1 = 'file_sha1'
FIELD_TOTAL_HASHES = 'total_hashes'

# TABLE FINGERPRINTS
FINGERPRINTS_TABLENAME = "fingerprints"

# FINGERPRINTS FIELDS
FIELD_HASH = 'hash'
FIELD_OFFSET = 'offset'

# FINGERPRINTS CONFIG:
# This is used as connectivity parameter for scipy.generate_binary_structure function. This parameter
# changes the morphology mask when looking for maximum peaks on the spectrogram matrix.
# Possible values are: [1, 2]
# Where 1 sets a diamond morphology which implies that diagonal elements are not considered as neighbors (this
# is the value used in the original dejavu code).
# And 2 sets a square mask, i.e. all elements are considered neighbors.
CONNECTIVITY_MASK = 2

# Sampling rate, related to the Nyquist conditions, which affects
# the range frequencies we can detect.
DEFAULT_FS = 44100

# Size of the FFT window, affects frequency granularity
DEFAULT_WINDOW_SIZE = 4096

# Ratio by which each sequential window overlaps the last and the
# next window. Higher overlap will allow a higher granularity of offset
# matching, but potentially more fingerprints.
DEFAULT_OVERLAP_RATIO = 0.5

# Degree to which a fingerprint can be paired with its neighbors. Higher values will
# cause more fingerprints, but potentially better accuracy.
DEFAULT_FAN_VALUE = 5  # 15 was the original value.

# Minimum amplitude in spectrogram in order to be considered a peak.
# This can be raised to reduce number of fingerprints, but can negatively
# affect accuracy.
DEFAULT_AMP_MIN = 10

# Number of cells around an amplitude peak in the spectrogram in order
# for Dejavu to consider it a spectral peak. Higher values mean less
# fingerprints and faster matching, but can potentially affect accuracy.
PEAK_NEIGHBORHOOD_SIZE = 10  # 20 was the original value.

# Thresholds on how close or far fingerprints can be in time in order
# to be paired as a fingerprint. If your max is too low, higher values of
# DEFAULT_FAN_VALUE may not perform as expected.
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200

# If True, will sort peaks temporally for fingerprinting;
# not sorting will cut down number of fingerprints, but potentially
# affect performance.
PEAK_SORT = True

# Number of bits to grab from the front of the SHA1 hash in the
# fingerprint calculation. The more you grab, the more memory storage,
# with potentially lesser collisions of matches.
FINGERPRINT_REDUCTION = 20

# Number of results being returned for file recognition
TOPN = 2



def unique_hash(file_path: str, block_size: int = 2**20) -> str:
    """ Small function to generate a hash to uniquely generate
    a file. Inspired by MD5 version here:
    http://stackoverflow.com/a/1131255/712997

    Works with large files.

    :param file_path: path to file.
    :param block_size: read block size.
    :return: a hash in an hexagesimal string form.
    """
    s = sha1()
    with open(file_path, "rb") as f:
        while True:
            buf = f.read(block_size)
            if not buf:
                break
            s.update(buf)
    return s.hexdigest().upper()


def find_files(path: str, extensions: List[str]) -> List[Tuple[str, str]]:
    """
    Get all files that meet the specified extensions.

    :param path: path to a directory with audio files.
    :param extensions: file extensions to look for.
    :return: a list of tuples with file name and its extension.
    """
    # Allow both with ".mp3" and without "mp3" to be used for extensions
    extensions = [e.replace(".", "") for e in extensions]

    results = []
    for dirpath, dirnames, files in os.walk(path):
        for extension in extensions:
            for f in fnmatch.filter(files, f"*.{extension}"):
                p = os.path.join(dirpath, f)
                results.append((p, extension))
    return results


def read(file_name: str, limit: int = None) -> Tuple[List[List[int]], int, str]:
    """
    Reads any file supported by pydub (ffmpeg) and returns the data contained
    within. If file reading fails due to input being a 24-bit wav file,
    wavio is used as a backup.

    Can be optionally limited to a certain amount of seconds from the start
    of the file by specifying the `limit` parameter. This is the amount of
    seconds from the start of the file.

    :param file_name: file to be read.
    :param limit: number of seconds to limit.
    :return: tuple list of (channels, sample_rate, content_file_hash).
    """
    # pydub does not support 24-bit wav files, use wavio when this occurs
    try:
        audiofile = AudioSegment.from_file(file_name)

        if limit:
            audiofile = audiofile[:limit * 1000]

        data = np.fromstring(audiofile.raw_data, np.int16)

        channels = []
        for chn in range(audiofile.channels):
            channels.append(data[chn::audiofile.channels])

        audiofile.frame_rate
    except audioop.error:
        _, _, audiofile = wavio.readwav(file_name)

        if limit:
            audiofile = audiofile[:limit * 1000]

        audiofile = audiofile.T
        audiofile = audiofile.astype(np.int16)

        channels = []
        for chn in audiofile:
            channels.append(chn)

    return channels, audiofile.frame_rate, unique_hash(file_name)


def get_audio_name_from_path(file_path: str) -> str:
    """
    Extracts song name from a file path.

    :param file_path: path to an audio file.
    :return: file name
    """
    return os.path.splitext(os.path.basename(file_path))[0]

def fingerprint(channel_samples: List[int],
                Fs: int = DEFAULT_FS,
                wsize: int = DEFAULT_WINDOW_SIZE,
                wratio: float = DEFAULT_OVERLAP_RATIO,
                fan_value: int = DEFAULT_FAN_VALUE,
                amp_min: int = DEFAULT_AMP_MIN) -> List[Tuple[str, int]]:
    """
    FFT the channel, log transform output, find local maxima, then return locally sensitive hashes.

    :param channel_samples: channel samples to fingerprint.
    :param Fs: audio sampling rate.
    :param wsize: FFT windows size.
    :param wratio: ratio by which each sequential window overlaps the last and the next window.
    :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list of hashes with their corresponding offsets.
    """
    # FFT the signal and extract frequency components
    arr2D = mlab.specgram(
        channel_samples,
        NFFT=wsize,
        Fs=Fs,
        window=mlab.window_hanning,
        noverlap=int(wsize * wratio))[0]

    # Apply log transform since specgram function returns linear array. 0s are excluded to avoid np warning.
    arr2D = 10 * np.log10(arr2D, out=np.zeros_like(arr2D), where=(arr2D != 0))

    local_maxima = get_2D_peaks(arr2D, plot=False, amp_min=amp_min)

    # return hashes
    return generate_hashes(local_maxima, fan_value=fan_value)


def get_2D_peaks(arr2D: np.array, plot: bool = False, amp_min: int = DEFAULT_AMP_MIN)\
        -> List[Tuple[List[int], List[int]]]:
    """
    Extract maximum peaks from the spectogram matrix (arr2D).

    :param arr2D: matrix representing the spectogram.
    :param plot: for plotting the results.
    :param amp_min: minimum amplitude in spectrogram in order to be considered a peak.
    :return: a list composed by a list of frequencies and times.
    """
    # Original code from the repo is using a morphology mask that does not consider diagonal elements
    # as neighbors (basically a diamond figure) and then applies a dilation over it, so what I'm proposing
    # is to change from the current diamond figure to a just a normal square one:
    #       F   T   F           T   T   T
    #       T   T   T   ==>     T   T   T
    #       F   T   F           T   T   T
    # In my local tests time performance of the square mask was ~3 times faster
    # respect to the diamond one, without hurting accuracy of the predictions.
    # I've made now the mask shape configurable in order to allow both ways of find maximum peaks.
    # That being said, we generate the mask by using the following function
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generate_binary_structure.html
    struct = generate_binary_structure(2, CONNECTIVITY_MASK)

    #  And then we apply dilation using the following function
    #  http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.iterate_structure.html
    #  Take into account that if PEAK_NEIGHBORHOOD_SIZE is 2 you can avoid the use of the scipy functions and just
    #  change it by the following code:
    #  neighborhood = np.ones((PEAK_NEIGHBORHOOD_SIZE * 2 + 1, PEAK_NEIGHBORHOOD_SIZE * 2 + 1), dtype=bool)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # find local maxima using our filter mask
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D

    # Applying erosion, the dejavu documentation does not talk about this step.
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # Boolean mask of arr2D with True at peaks (applying XOR on both matrices).
    detected_peaks = local_max != eroded_background

    # extract peaks
    amps = arr2D[detected_peaks]
    freqs, times = np.where(detected_peaks)

    # filter peaks
    amps = amps.flatten()

    # get indices for frequency and time
    filter_idxs = np.where(amps > amp_min)

    freqs_filter = freqs[filter_idxs]
    times_filter = times[filter_idxs]

    if plot:
        # scatter of the peaks
        fig, ax = plt.subplots()
        ax.imshow(arr2D)
        ax.scatter(times_filter, freqs_filter)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()

    return list(zip(freqs_filter, times_filter))


def generate_hashes(peaks: List[Tuple[int, int]], fan_value: int = DEFAULT_FAN_VALUE) -> List[Tuple[str, int]]:
    """
    Hash list structure:
       sha1_hash[0:FINGERPRINT_REDUCTION]    time_offset
        [(e05b341a9b77a51fd26, 32), ... ]

    :param peaks: list of peak frequencies and times.
    :param fan_value: degree to which a fingerprint can be paired with its neighbors.
    :return: a list of hashes with their corresponding offsets.
    """
    # frequencies are in the first position of the tuples
    idx_freq = 0
    # times are in the second position of the tuples
    idx_time = 1

    if PEAK_SORT:
        peaks.sort(key=itemgetter(1))

    hashes = []
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):

                freq1 = peaks[i][idx_freq]
                freq2 = peaks[i + j][idx_freq]
                t1 = peaks[i][idx_time]
                t2 = peaks[i + j][idx_time]
                t_delta = t2 - t1

                if MIN_HASH_TIME_DELTA <= t_delta <= MAX_HASH_TIME_DELTA:
                    h = sha1(f"{str(freq1)}|{str(freq2)}|{str(t_delta)}".encode('utf-8'))

                    hashes.append((h.hexdigest()[0:FINGERPRINT_REDUCTION], t1))

    return hashes

def get_file_fingerprints(file_name: str, limit: int = None, print_output: bool = False):
        channels, fs, file_hash = read(file_name, limit)
        fingerprints = set()
        channel_amount = len(channels)
        for channeln, channel in enumerate(channels, start=1):
            if print_output:
                print(f"Fingerprinting channel {channeln}/{channel_amount} for {file_name}")

            hashes = fingerprint(channel, Fs=fs)

            if print_output:
                print(f"Finished channel {channeln}/{channel_amount} for {file_name}")

            fingerprints |= set(hashes)

        return fingerprints, file_hash

fings = get_file_fingerprints('f1-drive_short.wav')
ff = list(fings[0])
ff.sort(key=lambda x: x[1])
ff = list(map(lambda x: list(x), ff))

def match_rate(base: list[tuple[str, int]], comp: list[tuple[str, int]]) -> float:
    matching_base_chunks = []
    matching_comp_chunks = []
    for c in comp:
        base_index = next((idx for (idx, b) in enumerate(base) if b[0] == c[0]), -1)
        if base_index > -1:
            # save all the matching base chunks
            # print(f"matching {c[0]} at {c[1]} to {base_index}")
            matching_base_chunks.append(base[base_index])
            matching_comp_chunks.append(c)

    return float(len(matching_base_chunks)) / float(len(comp))  # with this, i get far too high matches on the other
    # audio file

    # For each of them, make sure they are in the right order
    matching_base_chunks.sort(key=lambda x: x[1])  # sort the matching by time
    # compare if they are in the same order as the new track
    matching_chunks = float(0)
    for i in range(len(matching_comp_chunks)):
        if matching_comp_chunks[i][0] == matching_base_chunks[i][0]:
            matching_chunks += 1

    return matching_chunks / float(len(comp))

# def align_matches(self, matches: List[Tuple[int, int]], dedup_hashes: Dict[str, int], queried_hashes: int,
#                   topn: int = TOPN) -> List[Dict[str, any]]:
#     """
#     Finds hash matches that align in time with other matches and finds
#     consensus about which hashes are "true" signal from the audio.
#
#     :param matches: matches from the database
#     :param dedup_hashes: dictionary containing the hashes matched without duplicates for each song
#     (key is the song id).
#     :param queried_hashes: amount of hashes sent for matching against the db
#     :param topn: number of results being returned back.
#     :return: a list of dictionaries (based on topn) with match information.
#     """
#     # count offset occurrences per song and keep only the maximum ones.
#     sorted_matches = sorted(matches, key=lambda m: (m[0], m[1]))
#     counts = [(*key, len(list(group))) for key, group in groupby(sorted_matches, key=lambda m: (m[0], m[1]))]
#     songs_matches = sorted(
#         [max(list(group), key=lambda g: g[2]) for key, group in groupby(counts, key=lambda count: count[0])],
#         key=lambda count: count[2], reverse=True
#     )
#
#     songs_result = []
#     for song_id, offset, _ in songs_matches[0:topn]:  # consider topn elements in the result
#         song = self.db.get_song_by_id(song_id)
#
#         song_name = song.get(SONG_NAME, None)
#         song_hashes = song.get(FIELD_TOTAL_HASHES, None)
#         nseconds = round(float(offset) / DEFAULT_FS * DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO, 5)
#         hashes_matched = dedup_hashes[song_id]
#
#         song = {
#             SONG_ID: song_id,
#             SONG_NAME: song_name.encode("utf8"),
#             INPUT_HASHES: queried_hashes,
#             FINGERPRINTED_HASHES: song_hashes,
#             HASHES_MATCHED: hashes_matched,
#             # Percentage regarding hashes matched vs hashes from the input.
#             INPUT_CONFIDENCE: round(hashes_matched / queried_hashes, 2),
#             # Percentage regarding hashes matched vs hashes fingerprinted in the db.
#             FINGERPRINTED_CONFIDENCE: round(hashes_matched / song_hashes, 2),
#             OFFSET: offset,
#             OFFSET_SECS: nseconds,
#             FIELD_FILE_SHA1: song.get(FIELD_FILE_SHA1, None).encode("utf8")
#         }
#
#         songs_result.append(song)
#
#     return songs_result

other_fings = get_file_fingerprints('f1-middle.wav')
of = list(other_fings[0])
of = list(map(lambda x: list(x), of))
print(match_rate(ff, of))

ff.sort(key=lambda x: (x[1], x[0]))
of.sort(key=lambda x: (x[1], x[0]))
print(ff[:10])
print(of[:10])

indx = []
for i in range(len(of)):
    ff_ind = next((idx for (idx, b) in enumerate(ff) if b[0] == of[i][0]), -1)
    if ff_ind > -1:
        indx.append(ff_ind)
# print(sorted(indx))
print(len(of), len(indx))