import * as FileSystem from 'expo-file-system/legacy';

const root = FileSystem.documentDirectory + 'pitch-replay/';

export async function loadSessions() {
  const info = await FileSystem.getInfoAsync(root);
  if (!info.exists) return [];
  const names = await FileSystem.readDirectoryAsync(root);
  const sessions = [];
  for (const name of names.filter(name => name.endsWith('.json'))) {
    const session = JSON.parse(await FileSystem.readAsStringAsync(root + name));
    if ((await FileSystem.getInfoAsync(session.uri)).exists) sessions.push(session);
  }
  return sessions.sort((a,b) => b.created-a.created);
}

export async function saveSession(session) {
  await FileSystem.makeDirectoryAsync(root,{intermediates:true});
  await FileSystem.writeAsStringAsync(root + session.id + '.json',JSON.stringify(session));
}

export async function keepVideo(id, path) {
  await FileSystem.makeDirectoryAsync(root,{intermediates:true});
  const uri = root + id + '.mp4';
  await FileSystem.copyAsync({from:path.startsWith('file:') ? path : 'file://' + path,to:uri});
  return uri;
}
