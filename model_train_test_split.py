# train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)
print(X_train.shape, X_val.shape, X_test.shape)

# model

seq_len, feat_dim = X_train.shape[1], X_train.shape[2]
X_train = X_train.reshape(-1, seq_len, int(np.sqrt(feat_dim)), int(np.sqrt(feat_dim)), 1)
X_val   = X_val.reshape(-1,   seq_len, int(np.sqrt(feat_dim)), int(np.sqrt(feat_dim)), 1)
X_test  = X_test.reshape(-1,  seq_len, int(np.sqrt(feat_dim)), int(np.sqrt(feat_dim)), 1)

model = Sequential([
    ConvLSTM2D(32, (3,3), activation='relu', return_sequences=True, input_shape=X_train.shape[1:]),
    Dropout(0.2),
    TimeDistributed(MaxPooling2D((2,2))),
    ConvLSTM2D(64, (3,3), activation='relu', return_sequences=False),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.0005), metrics=['accuracy'])
model.summary()

#model performance metrics

history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=300, batch_size=4)
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NF','Violent'], yticklabels=['NF','Violent'])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.show()

