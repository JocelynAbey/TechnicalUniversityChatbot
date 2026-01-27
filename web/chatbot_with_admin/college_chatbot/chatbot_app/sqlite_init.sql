CREATE TABLE IF NOT EXISTS admin (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    password TEXT NOT NULL,
    typ TEXT NOT NULL,
    st INTEGER NOT NULL
);

INSERT INTO admin (username, password, typ, st)
SELECT 'admin', 'admin', 'admin', 1
WHERE NOT EXISTS (SELECT 1 FROM admin WHERE username = 'admin');

CREATE TABLE IF NOT EXISTS dataset_uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    upload_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_training (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_id TEXT NOT NULL,
    training_date TEXT NOT NULL,
    status INTEGER NOT NULL
);