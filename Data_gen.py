
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import skew, kurtosis, entropy as shannon_entropy
import tensorflow as tf
from gain import GAIN
from save_load import *

def approximate_entropy(U, m=2, r=0.2):
    def _phi(m):
        x = np.array([U[i:i + m] for i in range(len(U) - m + 1)])
        if len(x) == 0:
            return 0.0
        C = np.sum(
            np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0
        ) / (len(U) - m + 1 + 1e-10)
        C = np.clip(C, 1e-10, None)
        return np.sum(np.log(C)) / (len(U) - m + 1 + 1e-10)
    return abs(_phi(m) - _phi(m + 1))



# --- NSL-KDD processing ---
def NSL_KDD_PCA_features():
   #Data collection
    file_path = r"H:\vanisree\july  2025\Chowdary paper 2\Sourcecode\Dataset\NSL_KDD\KDDTrain+.txt"
    df = pd.read_csv(file_path, header=None)
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'level'
    ]
    df.columns = columns


    print(df)
    print(df.shape)
    # Data Preprocessing
    dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
    probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    r2l_attacks = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster', 'xclock', 'xsnoop']
    u2r_attacks = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm', 'http_tunnel', 'sendmail', 'named', 'snmpgetattack', 'snmpguess']

    def map_attack(attack):
        if attack in dos_attacks:
            return 1
        elif attack in probe_attacks:
            return 2
        elif attack in r2l_attacks:
            return 3
        elif attack in u2r_attacks:
            return 4
        else:
            return 0

    df['attack_map'] = df['attack'].apply(map_attack)

    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    numeric_df = df.select_dtypes(include=[np.number]).astype(float)
     #imputation using Generative Adversarial ImputationNetworks (GAIN)

    if numeric_df.isnull().values.any():
        print("NSL-KDD: Missing values — applying GAIN...")
        gain_model = GAIN(data_dim=numeric_df.shape[1], epochs=100)
        imputed = gain_model.fit_transform(numeric_df.values)
        numeric_df = pd.DataFrame(imputed, columns=numeric_df.columns)
    else:
        print("NSL-KDD: No missing values")

    numeric_df = (numeric_df - numeric_df.mean()) / numeric_df.std()
    numeric_df.fillna(0, inplace=True)
    #FeatureEngineering
    stats = pd.DataFrame()
    stats['mean'] = numeric_df.mean(axis=1)
    stats['var'] = numeric_df.var(axis=1)
    stats['skew'] = skew(numeric_df.values, axis=1, nan_policy='omit')
    stats['kurt'] = kurtosis(numeric_df.values, axis=1, nan_policy='omit')

    shannon = np.array([shannon_entropy(row + 1e-6) for row in numeric_df.values])
    approx = np.array([approximate_entropy(row + 1e-6) for row in numeric_df.values])
    entropy_df = pd.DataFrame({'shannon': shannon, 'approx': approx})

    input_dim = numeric_df.shape[1]
    ae_input = tf.keras.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(32, activation='relu')(ae_input)
    encoded = tf.keras.layers.Dense(16, activation='relu')(encoded)
    bottleneck = tf.keras.layers.Dense(8, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(16, activation='relu')(bottleneck)
    decoded = tf.keras.layers.Dense(32, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = tf.keras.Model(ae_input, decoded)
    encoder = tf.keras.Model(ae_input, bottleneck)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(numeric_df.values, numeric_df.values, epochs=50, batch_size=128, verbose=0)

    ae_features = encoder.predict(numeric_df.values)
    ae_df = pd.DataFrame(ae_features, columns=[f'AE_{i+1}' for i in range(ae_features.shape[1])])

    hybrid = pd.concat([stats, entropy_df, ae_df], axis=1)

    # --- Clean hybrid ---
    hybrid.replace([np.inf, -np.inf], np.nan, inplace=True)
    hybrid.fillna(0, inplace=True)
    hybrid = hybrid.clip(lower=-1e6, upper=1e6)
    #reducing dimensionality
    pca = PCA(n_components=0.95)
    reduced = pca.fit_transform(hybrid)

    reduced_df = pd.DataFrame(reduced, columns=[f'PC_{i+1}' for i in range(reduced.shape[1])])
    reduced_df['label'] = df['attack_map'].values

    return reduced_df

# --- CICIDS2017 processing ---
def CICIDS_PCA_features():
    # Data collection
    files = [
        r"Dataset\CICIDS2017\Benign-Monday-no-metadata.parquet",
        r"Dataset\CICIDS2017\Botnet-Friday-no-metadata.parquet"
    ]
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"CICIDS: Combined dataset shape: {df.shape}")

    df = df[:1000]
    # Data Preprocessing
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    numeric_df = df.select_dtypes(include=[np.number]).astype(float)
    # imputation using Generative Adversarial ImputationNetworks (GAIN)
    if numeric_df.isnull().values.any():
        print("CICIDS: Missing values — applying GAIN...")
        gain_model = GAIN(data_dim=numeric_df.shape[1], epochs=100)
        imputed = gain_model.fit_transform(numeric_df.values)
        numeric_df = pd.DataFrame(imputed, columns=numeric_df.columns)
    else:
        print("CICIDS: No missing values")

    numeric_df = (numeric_df - numeric_df.mean()) / numeric_df.std()
    numeric_df.fillna(0, inplace=True)
    # FeatureEngineering
    stats = pd.DataFrame()
    stats['mean'] = numeric_df.mean(axis=1)
    stats['var'] = numeric_df.var(axis=1)
    stats['skew'] = skew(numeric_df.values, axis=1, nan_policy='omit')
    stats['kurt'] = kurtosis(numeric_df.values, axis=1, nan_policy='omit')

    shannon = np.array([shannon_entropy(row + 1e-6) for row in numeric_df.values])
    approx = np.array([approximate_entropy(row + 1e-6) for row in numeric_df.values])
    entropy_df = pd.DataFrame({'shannon': shannon, 'approx': approx})

    input_dim = numeric_df.shape[1]
    ae_input = tf.keras.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(32, activation='relu')(ae_input)
    encoded = tf.keras.layers.Dense(16, activation='relu')(encoded)
    bottleneck = tf.keras.layers.Dense(8, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(16, activation='relu')(bottleneck)
    decoded = tf.keras.layers.Dense(32, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(input_dim, activation='linear')(decoded)

    autoencoder = tf.keras.Model(ae_input, decoded)
    encoder = tf.keras.Model(ae_input, bottleneck)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(numeric_df.values, numeric_df.values, epochs=50, batch_size=128, verbose=0)

    ae_features = encoder.predict(numeric_df.values)
    ae_df = pd.DataFrame(ae_features, columns=[f'AE_{i+1}' for i in range(ae_features.shape[1])])

    hybrid = pd.concat([stats, entropy_df, ae_df], axis=1)

    # --- Clean hybrid ---
    hybrid.replace([np.inf, -np.inf], np.nan, inplace=True)
    hybrid.fillna(0, inplace=True)
    hybrid = hybrid.clip(lower=-1e6, upper=1e6)
    # reducing dimensionality
    pca = PCA(n_components=0.95)
    reduced = pca.fit_transform(hybrid)

    reduced_df = pd.DataFrame(reduced, columns=[f'PC_{i+1}' for i in range(reduced.shape[1])])
    reduced_df['label'] = df[df.columns[-1]].values

    return reduced_df

#Multisource datafusion
def Datagen():
    nsl_df = NSL_KDD_PCA_features()
    cic_df = CICIDS_PCA_features()

    # Determine max PCs for alignment
    max_pcs = max(nsl_df.shape[1] - 1, cic_df.shape[1] - 1)

    # Enforce min 35, max 40 PCs
    target_pcs = max(35, min(max_pcs, 40))

    def pad_or_trim(df, target_pcs):
        pcs = [col for col in df.columns if col.startswith('PC_')]
        num_pcs = len(pcs)

        # Pad with zero columns if less than target
        if num_pcs < target_pcs:
            for i in range(num_pcs + 1, target_pcs + 1):
                df[f'PC_pad_{i}'] = 0.0
        # Trim if more than target
        elif num_pcs > target_pcs:
            keep_cols = pcs[:target_pcs]
            df = df[keep_cols + ['label']]
        return df

    nsl_df = pad_or_trim(nsl_df, target_pcs)
    cic_df = pad_or_trim(cic_df, target_pcs)

    # Combine
    combined = pd.concat([nsl_df, cic_df], ignore_index=True)

    # Split
    X = combined.drop(columns=['label']).values
    y = combined['label'].values

    print(f"Final feature dimension: {X.shape[1]}")  # Should be between 35-40

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    save("X_train_80", X_train)
    save("X_test_20", X_test)
    save("y_train_80", y_train)
    save("y_test_20", y_test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    save("X_train_70", X_train)
    save("X_test_30", X_test)
    save("y_train_70", y_train)
    save("y_test_30", y_test)
