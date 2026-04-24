import numpy as np
import matplotlib.pyplot as plt 
from numpy import sin, cos, pi
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def plot_12_lead_ecg(smaple_index, df, X, sampling_rate=100, percent=1):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    signal_array=X[smaple_index]
    signal_array = signal_array[:int(percent*signal_array.shape[0])]
    metadata_row=df.iloc[smaple_index]
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


    time = np.arange(signal_array.shape[0]) / sampling_rate
    bg_color = '#ffffff'
    line_color = '#000000'
    major_grid_color = '#ff9999'
    minor_grid_color = '#ffcccc'

    fig, axes = plt.subplots(12, 1, figsize=(15, 18), sharex=True)
    fig.patch.set_facecolor(bg_color)


    labels = metadata_row['diagnostic_superclass']
    patient_age = metadata_row['age']
    patient_sex = metadata_row['sex']

    fig.suptitle(f"ECG for Patient {metadata_row.name} | Age: {patient_age}, Sex: {patient_sex}\nDiagnostics: {labels}",
                 fontsize=16, y=0.92, color='black')

    for i in range(12):
        axes[i].set_facecolor(bg_color)
        axes[i].plot(time, signal_array[:, i], color=line_color, linewidth=2)

        axes[i].set_ylabel(leads[i], fontsize=12, rotation=0, labelpad=20, va='center', color='black')


        axes[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        axes[i].yaxis.set_minor_locator(AutoMinorLocator(5))

        axes[i].grid(True, which='major', color=major_grid_color, alpha=0.8, linewidth=1.0)
        axes[i].grid(True, which='minor', color=minor_grid_color, alpha=0.5, linewidth=0.5)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_color('black')
        axes[i].spines['left'].set_color('black')
        axes[i].tick_params(colors='black')

    axes[-1].set_xlabel("Time (Seconds)", fontsize=14, color='black')
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def plot_single_signal(signal:np.ndarray, sampling_rate=100, color='black', **kwargs):    
    assert len(signal.shape)==1

    time = np.arange(signal.shape[0]) / sampling_rate
    plt.plot(time, signal, linewidth=1, color=color, **kwargs)



def get_hann_weight(window_size=128):
    n = np.arange(window_size)
    w = 0.5 * (1 - np.cos((2 * np.pi * n) / (window_size - 1)))
    return w


def apply_bandpass_filter(signal, lowcut=0.5, highcut=45.0, fs=100.0, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def generate_ecg_spectrogram(x, window_size=128, max_freq=45, hop_size=4):
    time_steps = (len(x) - window_size) // hop_size + 1
    X = np.zeros((max_freq, time_steps), dtype='float64')
    
    n = np.arange(window_size)
    w = get_hann_weight(window_size)
    
    angles = np.zeros((max_freq, window_size))
    for omega in range(max_freq):
        angles[omega, :] = (omega * 2 * np.pi * n) / window_size
        
    cos_matrix = np.cos(angles)
    sin_matrix = -np.sin(angles)

    for step in range(time_steps):
        shift = step * hop_size
        chunk = x[shift:shift + window_size] * w
        
        a = np.dot(cos_matrix, chunk)
        b = np.dot(sin_matrix, chunk)
        
        X[:, step] = np.sqrt(a**2 + b**2)
        
    X_db = 20 * np.log10(X + 1e-10)
    
    db_min = np.min(X_db)
    db_max = np.max(X_db)
    
    if db_max > db_min:
        X_norm = (X_db - db_min) / (db_max - db_min) * 255.0
    else:
        X_norm = np.zeros_like(X_db)
        
    return X_norm.astype('uint8')


def generate_3d_tensor(patient_profile):
    result = []
    for i in range(12): # loop through the 12 leads
        x = patient_profile[:, i]
        x_clean = apply_bandpass_filter(x)
        spectrogram_tensor = generate_ecg_spectrogram(x_clean)
        result.append(spectrogram_tensor)

    return np.array(result)


def plot_before_after_transformation(
    signal,
    patient_idx,
    lead,
    sampling_rate=100,
    window_size=128,
    max_freq=45,
    hop_size=4,
    **kwargs,
):
    signal = np.asarray(signal)
    assert signal.ndim == 1, 'signal must be a 1D array'

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    plt.sca(axes[0])
    plot_single_signal(signal, sampling_rate=sampling_rate, **kwargs)
    axes[0].set_title(f'Original Signal from Lead "{lead}" for Patient "{patient_idx}"')
    axes[0].set_xlabel('Time (Seconds)')
    axes[0].set_ylabel('Amplitude')

    spectrogram = generate_ecg_spectrogram(
        signal,
        window_size=window_size,
        max_freq=max_freq,
        hop_size=hop_size,
    )
    axes[1].imshow(spectrogram, aspect='auto', origin='lower', cmap='jet', interpolation='bilinear')
    axes[1].set_title('Spectrogram After Applying STFT')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Frequency Bins')

    plt.tight_layout()
    plt.savefig('images/before_after.png')
    plt.show()


def normalize_signal(signal):
    normalized_signal = signal.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit_transform(normalized_signal)
    return np.array(normalized_signal.flatten())

def clean(signal_array:np.ndarray, bandpass_range=[-np.inf, np.inf], normalize=False, bandpass_order=3):
    processed = np.zeros_like(signal_array)

    for i in tqdm(range(signal_array.shape[0]), desc="Cleaning Signal"):
        for lead in range(12):
            patient_signal = signal_array[i][:, lead]

            lowcut, highcut=bandpass_range
            cleaned = apply_bandpass_filter(patient_signal, lowcut=lowcut, highcut=highcut, order=bandpass_order)

            if normalize:
                cleaned = normalize_signal(cleaned)

            processed[i][:, lead] = cleaned

    return processed