import numpy as np
from scipy.io import wavfile
from hashlib import sha256

sample_rate, base_data = wavfile.read('f1-drive_short.wav')
print('sample rate', sample_rate)


def extract_peak_frequency(data, sampling_rate):
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1 / sampling_rate)

    peak_coefficient = np.argmax(np.abs(fft_data))
    peak_freq = freqs[peak_coefficient]

    return abs(peak_freq * sampling_rate)


def extract_peak_frequency_in_range(data, sampling_rate, low, high):
    """
    :param data:
    :param sampling_rate:
    :param low:
    :param high:
    :return: the frequency, and the sample rate index at which it occurred
    """
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1 / sampling_rate)  # Ensure frequency calculation respects the sampling rate

    # Find indices where the frequency is between low and high
    indices = np.where((freqs >= low) & (freqs <= high))[0]

    # Subset the fft_data and freqs arrays
    fft_data_subset = fft_data[indices]
    freqs_subset = freqs[indices]

    # Find the peak in the specified frequency range
    peak_coefficient = np.argmax(np.abs(fft_data_subset))
    peak_freq = freqs_subset[peak_coefficient]

    return abs(peak_freq), peak_coefficient  # the index in the sample


# Break into chunks
ranges = [40, 80, 120, 180, 300, 400, 500, 800]


def get_file_peaks(file_path: str) -> list[tuple[int, int, int, int]]:
    sample_rate, base_data = wavfile.read(file_path)
    chunk_size = int(sample_rate / 10)
    num_chunks = int(len(base_data) / chunk_size)
    base_chunks = []
    for i in range(num_chunks - 1):
        base_chunks.append(base_data[chunk_size * i:chunk_size * (i + 1)])
    print(len(base_chunks), "total chunks")
    peak, ind = extract_peak_frequency_in_range(base_chunks[1],
                                                sample_rate, 100,
                                                500)
    print(peak, ind + chunk_size, "second chunk peak")  # find the peak of the range, offset from the beginning

    # For each chunk, get the peak frequencies from each range
    peaks = []
    for chunk_i in range(len(base_chunks)):
        chunk = base_chunks[chunk_i]
        for i in range(len(ranges) - 1):
            peak, offset = extract_peak_frequency_in_range(chunk, sample_rate, ranges[i], ranges[i + 1])
            fuz_peak = int(peak) - (int(peak) % 2)  # floor to power of 2
            # print(f"peak base data {ranges[i]}-{ranges[i + 1]}", fuz_peak, offset)
            peaks.append((fuz_peak, offset + chunk_size * chunk_i, ranges[i], ranges[i + 1]))

    return peaks


def hash_peaks(peaks: list[tuple[int, int]], spread: int, min_time_delta: int = 1) -> list[tuple[str, int]]:
    """
    Hashes peaks
    :param peaks: List of peak, time from start of file
    :param spread: How many peaks forward in the list will be considered for building a hash.
    :param min_time_delta: The minimum amount of time that a neighbor can be considered at.
    :return: List of hashes and their timestamps (t1)
    """

    # Ensure that the peaks are sorted by time
    peaks.sort(key=lambda x: x[1])

    hashes = []
    for i in range(len(peaks)):
        for j in range(1, spread):
            if (i + j) < len(peaks):
                f1 = peaks[i][0]
                f2 = peaks[i + j][0]
                t1 = peaks[i][1]
                t2 = peaks[i + j][1]
                t_delta = t2 - t1
                if t_delta < min_time_delta:
                    # Not enough of a delta
                    continue

                # print('hashing', f1, f2, t1, t2)
                h = sha256(f"{str(f1)}::{str(f2)}::{str(t_delta)}".encode("utf-8"))
                hashes.append((h.hexdigest(), t1))

    return hashes


base_peaks = get_file_peaks('f1-drive_short.wav')
base_hashes = hash_peaks(base_peaks, 6)
print(base_hashes[:10])
print(len(base_hashes))
base_hash_list = list(map(lambda x: x[0], base_hashes))

# Compare a small slice
small_peaks = get_file_peaks('f1-drive_short-slice.wav')
small_hashes = hash_peaks(small_peaks, 6)
print(small_hashes[:10])
print(len(small_hashes))
small_hash_list = list(map(lambda x: x[0], small_hashes))

matching = 0
for hash in small_hash_list:
    if hash in base_hash_list:
        matching += 1

print("small match:", float(matching) / float(len(small_hashes)))

# Compare an external recording
external_peaks = get_file_peaks('laptop-rec-bg-noise.wav')
external_hashes = hash_peaks(external_peaks, 6)
print(external_hashes[:10])
print(len(external_hashes))
external_hash_list = list(map(lambda x: x[0], external_hashes))

matching = 0
for hash in external_hash_list:
    if hash in base_hash_list:
        matching += 1

print("external match:", float(matching) / float(len(external_hashes)))