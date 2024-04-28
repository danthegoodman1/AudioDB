import numpy as np
from scipy.io import wavfile
from hashlib import sha256

import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.fft import fft, fftfreq


def extract_peak_frequency_in_range(data, sampling_rate, low, high):
    # Convert the data to float
    data = data.astype(float)

    # Perform FFT on the data
    fft_data = fft(data)

    # Calculate the frequencies corresponding to the FFT
    frequencies = fftfreq(len(data), 1 / sampling_rate)

    # Find the indices corresponding to the desired frequency range
    low_index = np.argmax(frequencies >= low)
    high_index = np.argmin(frequencies <= high)

    # Extract the FFT magnitudes within the desired frequency range
    fft_magnitudes = np.abs(fft_data[low_index:high_index])

    # Find the index of the peak frequency within the range
    peak_index = np.argmax(fft_magnitudes)

    # Calculate the actual peak frequency
    peak_frequency = frequencies[low_index + peak_index]

    # Calculate the time offset of the peak frequency
    peak_time_offset = (low_index + peak_index) / sampling_rate

    return peak_frequency, peak_time_offset


# Break into chunks
ranges = [40, 80, 120, 180, 300, 600, 800]
spread = 6


def get_file_peaks(file_path: str) -> list[tuple[int, int, int, int]]:
    print("reading file", file_path)
    sample_rate, base_data = wavfile.read(file_path)
    print('sample rate', sample_rate)
    chunk_size = int(sample_rate / 40)
    num_chunks = int(len(base_data) / chunk_size)
    base_chunks = []
    for i in range(num_chunks - 1):
        base_chunks.append(base_data[chunk_size * i:chunk_size * (i + 1)])
    print(len(base_chunks), "total chunks")

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


def plot_file_with_peaks(file_path: str):

    # Get peaks and data
    peaks = get_file_peaks(file_path)
    sample_rate, base_data = wavfile.read(file_path)

    # Compute the spectrogram of the audio data
    f, t, Sxx = spectrogram(base_data, sample_rate)

    # Create a new figure with specified size
    plt.figure(figsize=(12, 3))

    # Plot the spectrogram
    plt.pcolormesh(t, f, 10 * np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.ylim(0, 1000)
    plt.xlim(t.min(), t.max())  # fix weird white gap with scatter plot

    plt.savefig('before-peaks.png', dpi=300, bbox_inches='tight')
    for peak in peaks:
        plt.scatter(peak[1] / sample_rate, peak[0], color='r', s=25) # converting samples to seconds for the x coordinate

    # Save the figure to a file before displaying it.
    plt.savefig('with-peaks.png', dpi=300, bbox_inches='tight')


# plot_file_with_peaks("f1-drive_short.wav")


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

def match_rate(base: list[tuple[str, int]], comp: list[tuple[str, int]]) -> float:
    matching_base_chunks = []
    matching_comp_chunks = []
    for h in comp:
        base_index = next((idx for (idx, b) in enumerate(base) if b[0] == h[0]), -1)
        if base_index > -1:
            # save all the matching base chunks
            matching_base_chunks.append(base[base_index])  # hash and time so we can sort
            matching_comp_chunks.append(h[0])  # just the hash

    # return float(len(matching_base_chunks)) / float(len(comp))  # with this, i get far too high matches on the other
    # audio file

    # For each of them, make sure they are in the right order
    matching_base_chunks.sort(key=lambda x: x[1])  # sort the matching by time
    mbc_hashes = list(map(lambda x: x[0], matching_base_chunks))
    # compare if they are in the same order as the new track
    matching_chunks = float(0)
    last_comp_ind = 0
    for chunk in matching_comp_chunks:
        # Get the index of the base chunk and ensure it exists in ascending order
        try:
            mbc_ind = mbc_hashes.index(chunk)
            if mbc_ind > last_comp_ind:
                # print(matching_comp_chunks[i], mbc_hashes[mbc_hashes.index(matching_comp_chunks[i])])
                matching_chunks += 1
            last_comp_ind = mbc_ind
        except:
            pass

    return matching_chunks / float(len(comp))


base_peaks = get_file_peaks('f1-drive_short.wav')
base_hashes = hash_peaks(base_peaks, spread)
print("total hashes:", len(base_hashes))
base_hash_list = list(map(lambda x: x[0], base_hashes))

# Compare a small slice (this is small slice of the original file)
small_peaks = get_file_peaks('f1-start.wav')
small_hashes = hash_peaks(small_peaks, spread)
print("total hashes:", len(small_hashes))
small_hash_list = list(map(lambda x: x[0], small_hashes))

print("small start match:", match_rate(base_hashes, small_hashes))

middle_peaks = get_file_peaks('f1-middle.wav')
middle_hashes = hash_peaks(middle_peaks, spread)
print("total hashes:", len(middle_hashes))
middle_hash_list = list(map(lambda x: x[0], middle_hashes))

print("middle match:", match_rate(base_hashes, middle_hashes))

external_peaks = get_file_peaks('laptop-rec.wav')
external_hashes = hash_peaks(external_peaks, spread)
print("total hashes:", len(external_hashes))
external_hash_list = list(map(lambda x: x[0], external_hashes))

print("external match:", match_rate(base_hashes, external_hashes))

other_peaks = get_file_peaks('other-audio.wav')
other_hashes = hash_peaks(other_peaks, spread)
print("total hashes:", len(other_hashes))
other_hash_list = list(map(lambda x: x[0], other_hashes))


print("other audio noise match:", match_rate(base_hashes, other_hashes))