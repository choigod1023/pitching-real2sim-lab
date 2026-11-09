const opened = new Promise((resolve, reject) => {
  const request = indexedDB.open('pitch-replay', 1);
  request.onupgradeneeded = () => request.result.createObjectStore('sessions', {keyPath:'id'});
  request.onsuccess = () => resolve(request.result);
  request.onerror = () => reject(request.error);
});

export async function listSessions() {
  const db = await opened;
  return new Promise((resolve, reject) => {
    const request = db.transaction('sessions').objectStore('sessions').getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveSession(session) {
  const db = await opened;
  await new Promise((resolve, reject) => {
    const transaction = db.transaction('sessions', 'readwrite');
    transaction.objectStore('sessions').put(session);
    transaction.oncomplete = resolve;
    transaction.onabort = () => reject(transaction.error);
  });
}
