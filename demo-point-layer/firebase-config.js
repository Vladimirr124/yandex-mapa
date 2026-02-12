/**
 * Конфиг Firebase Realtime Database для синхронизации поездов между всеми, кто открыл карту по ссылке.
 * Если оставить null — поезда будут только у вас локально.
 * Чтобы включить общую карту: создайте проект на https://console.firebase.google.com/ ,
 * включите Realtime Database, скопируйте конфиг из «Настройки проекта → Ваши приложения» сюда.
 */
var FIREBASE_CONFIG = null;

/* Пример заполненного конфига (подставьте свои значения):
var FIREBASE_CONFIG = {
  apiKey: "AIza...",
  authDomain: "your-project.firebaseapp.com",
  databaseURL: "https://your-project-default-rtdb.europe-west1.firebasedatabase.app",
  projectId: "your-project",
  storageBucket: "your-project.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abc"
};
*/
