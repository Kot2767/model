import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import math
from collections import deque
import os
import pickle
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

# ------------------ Часть 1: Распознавание жестов через MediaPipe ------------------
class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        self.trajectory = deque(maxlen=10)      # история позиций центра ладони для свайпов
        self.center_history = deque(maxlen=20)  # для определения направления движения
        self.feature_mean = None   # для нормализации (будут вычислены на обучающих данных)
        self.feature_std = None

    def extract_features(self, hand_landmarks, handedness):
        """Извлекает признаки из landmarks руки: только координаты (63) + флаг руки"""
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        # флаг руки: 0 - правая, 1 - левая (или наоборот)
        hand_flag = 0 if handedness == 'Right' else 1
        features = np.array(coords + [hand_flag])  # теперь 64 признака
        return features

    def normalize_features(self, features):
        """Применяет нормализацию, если известны mean/std"""
        if self.feature_mean is not None and self.feature_std is not None:
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)
        return features

    def detect_swipe(self, center_history, dt=1/30):
        """
        Определяет свайп по истории положений центра ладони с учётом скорости.
        dt - время между кадрами (при 30 fps)
        """
        if len(center_history) < 5:
            return None
        arr = np.array(center_history)
        # скорость как разность последнего и первого
        delta = arr[-1] - arr[0]
        # грубая скорость (пикселей/кадр) - порог можно калибровать
        speed = np.linalg.norm(delta) / (len(center_history)-1)
        if speed > 0.03:   # эмпирический порог
            if abs(delta[0]) > abs(delta[1]):
                return "SWIPE_RIGHT" if delta[0] > 0 else "SWIPE_LEFT"
            else:
                return "SWIPE_DOWN" if delta[1] > 0 else "SWIPE_UP"
        return None

    def get_palm_center(self, hand_landmarks):
        """Вычисляет центр ладони (среднее всех landmarks)"""
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        return np.array([np.mean(xs), np.mean(ys)])

    def process_frame(self, frame):
        """Обрабатывает кадр: определяет руки и возвращает список признаков для каждой руки"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hands_data = []
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                   results.multi_handedness):
                # рисуем ключевые точки
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # извлекаем признаки
                features = self.extract_features(hand_landmarks, handedness.classification[0].label)
                # нормализуем
                features = self.normalize_features(features)
                # центр ладони
                center = self.get_palm_center(hand_landmarks)
                hands_data.append({
                    'landmarks': hand_landmarks,
                    'features': features,
                    'center': center,
                    'handedness': handedness.classification[0].label
                })
        return frame, hands_data

    def set_normalization_params(self, mean, std):
        """Устанавливает параметры нормализации (из обучающего набора)"""
        self.feature_mean = mean
        self.feature_std = std


# ------------------ Часть 2: Модели на TensorFlow/Keras ------------------
class GestureSequenceModel:
    """Модель для анализа последовательностей жестов (LSTM + Attention)"""
    def __init__(self, num_classes, seq_len=30, input_dim=64):
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.model = self.build_model()

    def build_model(self):
        inp = tf.keras.Input(shape=(self.seq_len, self.input_dim))
        # Bidirectional LSTM
        o = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)  # (batch, seq, 256)
        # Внимание
        u = layers.Dense(1, activation='tanh')(o)          # (batch, seq, 1)
        u = layers.Flatten()(u)                             # (batch, seq)
        u = layers.Activation('softmax')(u)                 # веса
        u = layers.RepeatVector(256)(u)                     # (batch, 256, seq)
        u = layers.Permute([2, 1])(u)                       # (batch, seq, 256)
        a = layers.Multiply()([o, u])                        # взвешенные состояния
        a = layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(a)  # (batch, 256)
        # Классификатор
        s = layers.Dense(256, activation='relu')(a)
        s = layers.Dropout(0.5)(s)
        s = layers.Dense(128, activation='relu')(s)
        s = layers.Dense(self.num_classes, activation='softmax')(s)
        model = tf.keras.Model(inp, s)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y, epochs=10, batch_size=32, validation_split=0.2):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)


class AdvancedGestureCNN:
    """CNN для классификации отдельных изображений жестов"""
    def __init__(self, num_classes, input_shape=(128,128,3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3,3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3,3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2,2)),
            layers.Dropout(0.25),

            layers.Conv2D(256, (3,3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),

            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, X, y, epochs=10, batch_size=32, validation_split=0.2):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)


# ------------------ Часть 3: Модели на PyTorch ------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0)]


class MultiHeadAttention(nn.Module):
    """Обёртка над встроенным MultiheadAttention для удобства"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch, seq_len, d_model)
        attn_out, attn_weights = self.attn(query, key, value, attn_mask=mask)
        return attn_out, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class GestureTransformer(nn.Module):
    """Трансформер для классификации последовательностей жестов (64 признака на кадр)"""
    def __init__(self, input_dim=64, d_model=256, num_heads=8, num_layers=6,
                 dim_feedforward=512, num_classes=29, max_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        # Классификатор с пулингом по времени
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        # Используем среднее по времени вместо последнего кадра
        x = x.mean(dim=1)  # (batch, d_model)
        return self.classifier(x)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))


class MultiScaleCNN(nn.Module):
    """Многомасштабная 1D свёртка для извлечения локальных признаков (для аудио)"""
    def __init__(self, in_channels, out_channels=512):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=2, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=4, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=6, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        ])

    def forward(self, x):
        # x: (batch, seq_len, in_channels)
        x = x.transpose(1, 2)  # (batch, in_channels, seq_len)
        out = []
        for conv in self.convs:
            y = conv(x)
            y = F.adaptive_max_pool1d(y, 1).squeeze(-1)  # (batch, out_channels)
            out.append(y)
        return torch.cat(out, dim=1)  # (batch, 3*out_channels)


class AdvancedLSTM(nn.Module):
    """LSTM с механизмом внимания (gating) для аудио/текста"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Linear(lstm_out_dim, lstm_out_dim)
        self.norm = nn.LayerNorm(lstm_out_dim)

    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size(0), x.size(1)
        if hidden is None:
            h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                              batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros_like(h0)
            hidden = (h0, c0)
        out, (hn, cn) = self.lstm(x, hidden)
        # attention gate
        gate = torch.sigmoid(self.attention(out))
        gated = out * gate
        out = self.norm(gated)
        return out, (hn, cn)


class TransformerBlock(nn.Module):
    """Блок трансформера для аудио, объединяющий MultiHeadAttention и MultiScaleCNN"""
    def __init__(self, d_model, num_heads, cnn_out=512, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cnn = MultiScaleCNN(d_model, cnn_out)
        self.proj_cnn = nn.Linear(3 * cnn_out, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Multi-scale CNN
        cnn_out = self.cnn(x)  # (batch, 3*cnn_out)
        cnn_out = self.proj_cnn(cnn_out).unsqueeze(1).expand(-1, x.size(1), -1)
        x = self.norm2(x + self.dropout(cnn_out))
        # Feed-forward
        ff_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x, attn_weights


# ------------------ Часть 4: Аудио процессор ------------------
class AudioProcessor:
    """Загружает аудио, извлекает MFCC, прогоняет через модели постобработки"""
    def __init__(self, device='cpu'):
        self.device = device
        self.lstm = AdvancedLSTM(input_size=13, hidden_size=256).to(device)   # 13 MFCC
        self.transformer_block = TransformerBlock(d_model=256, num_heads=8).to(device)
        self.classifier = nn.Linear(256, 29).to(device)  # предположим 29 классов (буквы/фонемы)
        self.models_loaded = False

    def load_audio(self, file_path, sr=16000, n_mfcc=13, max_len=200):
        """Загружает аудио и возвращает тензор MFCC (batch, time, n_mfcc)"""
        try:
            signal, sr = librosa.load(file_path, sr=sr)
        except Exception as e:
            print(f"Ошибка загрузки аудио: {e}")
            return None
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc).T  # (time, n_mfcc)
        # Приведение к фиксированной длине
        if mfcc.shape[0] < max_len:
            pad = max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_len, :]
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, time, 13)
        return tensor

    def forward(self, mfcc_tensor):
        """Прогоняет MFCC через модели и возвращает логиты"""
        lstm_out, _ = self.lstm(mfcc_tensor)  # (batch, time, 256)
        transformed, _ = self.transformer_block(lstm_out)  # (batch, time, 256)
        # Пулинг по времени (среднее)
        pooled = transformed.mean(dim=1)  # (batch, 256)
        logits = self.classifier(pooled)
        return logits

    def process_file(self, file_path):
        """Высокоуровневый метод: загрузка + прогон"""
        mfcc = self.load_audio(file_path)
        if mfcc is None:
            return None
        with torch.no_grad():
            logits = self.forward(mfcc)
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
        return pred, probs.cpu().numpy()

    def save(self, path):
        torch.save({
            'lstm': self.lstm.state_dict(),
            'transformer_block': self.transformer_block.state_dict(),
            'classifier': self.classifier.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.lstm.load_state_dict(checkpoint['lstm'])
        self.transformer_block.load_state_dict(checkpoint['transformer_block'])
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.models_loaded = True


# ------------------ Часть 5: Контекстный анализ (выбор буквы на основе предыдущих) ------------------
class ContextualModel(nn.Module):
    """Модель для оценки вероятности следующей буквы по предыдущим (простой LSTM)"""
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) индексы букв
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = out[:, -1, :]  # последний выход
        return self.fc(out)  # логиты для следующей буквы

    def get_bonus(self, prefix, candidates):
        """Возвращает дополнительный балл для каждой буквы-кандидата на основе prefix"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(prefix.unsqueeze(0))  # (1, vocab)
            probs = F.softmax(logits, dim=-1).squeeze(0)  # (vocab)
            bonus = probs[candidates].cpu().numpy()
        return bonus

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device))


# ------------------ Часть 6: Тренер для PyTorch моделей ------------------
class GestureTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-4)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.clip_grad_norm = 1.0

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        self.scheduler.step()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return correct / total


# ------------------ Часть 7: Основная интегрирующая система ------------------
class NeuroAdaptiveSystem:
    """
    Главный класс, объединяющий все компоненты системы.
    """
    def __init__(self, gesture_classes, vocab_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.gesture_classes = gesture_classes  # список названий жестов (включая свайпы)
        self.vocab_classes = vocab_classes      # список букв/слов для контекста
        self.num_gestures = len(gesture_classes)
        self.vocab_size = len(vocab_classes)

        # Инициализация компонентов
        self.recognizer = GestureRecognizer()

        # Модели TensorFlow (видео)
        self.cnn_model = AdvancedGestureCNN(self.num_gestures)
        self.lstm_model = GestureSequenceModel(self.num_gestures)

        # Модели PyTorch (видео + аудио)
        self.transformer = GestureTransformer(num_classes=self.num_gestures).to(device)
        self.context_model = ContextualModel(self.vocab_size).to(device)
        self.audio_processor = AudioProcessor(device)

        # Состояние для реального времени
        self.feature_buffer = deque(maxlen=30)  # буфер последних 30 кадров признаков
        self.last_gesture = None
        self.swipe_detected = False

        # Флаги загрузки
        self.models_loaded = False

    def load_models(self, cnn_path=None, lstm_path=None, transformer_path=None,
                    context_path=None, audio_path=None, norm_params_path=None):
        """Загружает веса обученных моделей и параметры нормализации"""
        if cnn_path and os.path.exists(cnn_path):
            self.cnn_model.load(cnn_path)
            print("CNN model loaded.")
        if lstm_path and os.path.exists(lstm_path):
            self.lstm_model.load(lstm_path)
            print("LSTM model loaded.")
        if transformer_path and os.path.exists(transformer_path):
            self.transformer.load(transformer_path, self.device)
            print("Transformer loaded.")
        if context_path and os.path.exists(context_path):
            self.context_model.load(context_path, self.device)
            print("Context model loaded.")
        if audio_path and os.path.exists(audio_path):
            self.audio_processor.load(audio_path)
            print("Audio processor loaded.")
        if norm_params_path and os.path.exists(norm_params_path):
            params = np.load(norm_params_path)
            self.recognizer.set_normalization_params(params['mean'], params['std'])
            print("Normalization params loaded.")
        self.models_loaded = True

    # ---------- Видео-обработка в реальном времени ----------
    def process_video_frame(self, frame):
        """Обрабатывает кадр, накапливает признаки, распознаёт жест и возвращает аннотированный кадр"""
        annotated, hands = self.recognizer.process_frame(frame)

        # Если есть руки, добавляем признаки в буфер
        if hands:
            # Для простоты берём первую руку (можно усреднить или обрабатывать отдельно)
            features = hands[0]['features']
            self.feature_buffer.append(features)
            # Детекция свайпа
            self.recognizer.center_history.append(hands[0]['center'])
            swipe = self.recognizer.detect_swipe(list(self.recognizer.center_history))
            if swipe:
                self.last_gesture = swipe
                self.swipe_detected = True
        else:
            # Если рук нет, очищаем буфер (или можно вставлять нули)
            self.feature_buffer.clear()
            self.recognizer.center_history.clear()
            self.swipe_detected = False

        # Если накоплено достаточно кадров, классифицируем последовательность
        if len(self.feature_buffer) == 30 and not self.swipe_detected:
            seq = np.array(self.feature_buffer)  # (30, 64)
            # Классификация несколькими моделями (голосование)
            preds = []
            # LSTM (TF)
            lstm_input = np.expand_dims(seq, 0)
            lstm_prob = self.lstm_model.predict(lstm_input)[0]
            preds.append(np.argmax(lstm_prob))

            # Transformer (PyTorch)
            tensor_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                trans_logits = self.transformer(tensor_seq)
                trans_prob = F.softmax(trans_logits, dim=-1).cpu().numpy()[0]
            preds.append(np.argmax(trans_prob))

            # CNN по ключевому кадру (среднему, например)
            # Здесь можно добавить использование CNN, если есть доступ к изображению
            # Пока пропустим

            # Голосование (самый частый класс)
            final_pred = max(set(preds), key=preds.count)
            self.last_gesture = self.gesture_classes[final_pred]
            # Можно применить контекстный бонус, если есть префикс
            # ...

        return annotated, self.last_gesture

    # ---------- Работа с видеофайлами ----------
    def classify_video(self, video_path):
        """Загружает видео, собирает последовательность и классифицирует"""
        seq = self.collect_gesture_sequence(video_path)
        if seq is None:
            return None
        # Используем трансформер для классификации
        tensor_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.transformer(tensor_seq)
            pred = torch.argmax(logits, dim=-1).item()
        return self.gesture_classes[pred]

    def collect_gesture_sequence(self, video_path, max_frames=30):
        """Собирает последовательность признаков из видео"""
        cap = cv2.VideoCapture(video_path)
        features_seq = []
        while len(features_seq) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            _, hands = self.recognizer.process_frame(frame)
            if hands:
                features_seq.append(hands[0]['features'])
            else:
                # если руки нет, добавляем нули (64)
                features_seq.append(np.zeros(64))
        cap.release()
        if len(features_seq) == 0:
            return None
        # обрезаем/дополняем до max_frames
        if len(features_seq) < max_frames:
            pad = max_frames - len(features_seq)
            features_seq.extend([np.zeros(64)] * pad)
        else:
            features_seq = features_seq[:max_frames]
        return np.array(features_seq)

    # ---------- Аудио-обработка ----------
    def process_audio_file(self, audio_path):
        """Загружает аудиофайл и возвращает предсказанный класс"""
        if not self.audio_processor.models_loaded:
            print("Аудиомодели не загружены.")
            return None
        pred, probs = self.audio_processor.process_file(audio_path)
        return pred, probs

    # ---------- Контекстный анализ ----------
    def contextual_boost(self, prefix_indices, candidate_indices):
        """
        Возвращает дополнительные баллы для кандидатов на основе предыдущих букв.
        prefix_indices: список индексов предыдущих букв (например, [2,5,1])
        candidate_indices: список индексов букв-кандидатов
        """
        prefix = torch.tensor(prefix_indices, dtype=torch.long).to(self.device)
        bonus = self.context_model.get_bonus(prefix, torch.tensor(candidate_indices))
        return bonus

    # ---------- Обучение новым жестам ----------
    def collect_training_data(self, gesture_name, mode='frame', num_samples=100, save_dir='data'):
        """
        Собирает данные для обучения новому жесту.
        mode: 'frame' - одиночные кадры, 'sequence' - видео (последовательность кадров)
        """
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(0)
        samples = []
        if mode == 'sequence':
            # Для последовательности будем записывать по 30 кадров при нажатии
            print(f"Сбор последовательностей для жеста '{gesture_name}'. Нажмите пробел для записи 30 кадров, 'q' для выхода.")
        else:
            print(f"Сбор кадров для жеста '{gesture_name}'. Нажмите пробел для захвата, 'q' для выхода.")
        while len(samples) < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, hands = self.recognizer.process_frame(frame)
            cv2.imshow('Collect', annotated)
            key = cv2.waitKey(1)
            if key == ord(' '):
                if hands:
                    if mode == 'frame':
                        samples.append({
                            'image': frame.copy(),
                            'features': hands[0]['features']
                        })
                        print(f"Сэмпл {len(samples)}/{num_samples}")
                    else:  # sequence
                        seq = []
                        # Собираем следующие 30 кадров (или пока не накопится)
                        for _ in range(30):
                            ret, f = cap.read()
                            if not ret:
                                break
                            _, h = self.recognizer.process_frame(f)
                            if h:
                                seq.append(h[0]['features'])
                            else:
                                seq.append(np.zeros(64))
                        if len(seq) == 30:
                            samples.append({
                                'sequence': np.array(seq),
                                'label': gesture_name
                            })
                            print(f"Последовательность {len(samples)}/{num_samples}")
                        else:
                            print("Не удалось собрать 30 кадров, пропуск.")
                else:
                    print("Рука не обнаружена")
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        # сохраняем
        with open(f"{save_dir}/{gesture_name}_{mode}.pkl", 'wb') as f:
            pickle.dump(samples, f)
        print(f"Данные сохранены в {save_dir}/{gesture_name}_{mode}.pkl")

    def train_new_gesture(self, gesture_name, model_type='cnn', epochs=10, data_dir='data'):
        """
        Обучает модель на собранных данных для нового жеста.
        model_type: 'cnn' (по изображениям), 'lstm' (по последовательностям), 'transformer'
        """
        if model_type == 'cnn':
            with open(f"{data_dir}/{gesture_name}_frame.pkl", 'rb') as f:
                samples = pickle.load(f)
            X = []
            y = []
            for s in samples:
                img = cv2.resize(s['image'], (128,128))
                X.append(img)
                y.append(self.gesture_classes.index(gesture_name))
            X = np.array(X) / 255.0
            y = tf.keras.utils.to_categorical(y, num_classes=self.num_gestures)
            self.cnn_model.model.fit(X, y, epochs=epochs, validation_split=0.2)
            self.cnn_model.save(f"models/cnn_{gesture_name}.h5")
        elif model_type in ['lstm', 'transformer']:
            with open(f"{data_dir}/{gesture_name}_sequence.pkl", 'rb') as f:
                samples = pickle.load(f)
            X = np.array([s['sequence'] for s in samples])
            y = np.array([self.gesture_classes.index(gesture_name) for _ in samples])
            if model_type == 'lstm':
                # Преобразуем для TF
                y_cat = tf.keras.utils.to_categorical(y, num_classes=self.num_gestures)
                self.lstm_model.model.fit(X, y_cat, epochs=epochs, validation_split=0.2)
                self.lstm_model.save(f"models/lstm_{gesture_name}.h5")
            else:  # transformer
                # Обучаем через PyTorch
                dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.long)
                )
                loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
                trainer = GestureTrainer(self.transformer, self.device)
                for epoch in range(epochs):
                    loss = trainer.train_epoch(loader)
                    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
                self.transformer.save(f"models/transformer_{gesture_name}.pth")
        else:
            raise ValueError("Unknown model type")


# ------------------ Часть 8: Пример использования ------------------
if __name__ == "__main__":
    # Предполагаемые классы жестов и алфавит
    gesture_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                      'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                      'SPACE', 'DELETE', 'SWIPE_LEFT', 'SWIPE_RIGHT', 'SWIPE_UP', 'SWIPE_DOWN']
    alphabet = list('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')  # для контекста

    system = NeuroAdaptiveSystem(gesture_labels, alphabet)

    # Загрузка предобученных весов (если есть)
    # system.load_models(cnn_path='models/cnn.h5', transformer_path='models/transformer.pth',
    #                    norm_params_path='models/norm_params.npz')

    # Запуск веб-камеры для демонстрации
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, gesture = system.process_video_frame(frame)
        if gesture:
            cv2.putText(annotated, f"Gesture: {gesture}", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('NeuroAdaptive System', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
