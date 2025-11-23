// tampingLogger.js
const fs = require("fs");
const path = require("path");

// Carpeta y archivo donde guardaremos el JSON
const logsDir = path.join(__dirname, "logs");
const logsFile = path.join(logsDir, "tamping_events.json");

// Asegurar que exista la carpeta logs
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

/**
 * Guarda un evento en logs/tamping_events.json
 * El archivo contiene un arreglo de objetos JSON.
 */
function saveTampingEvent(event) {
  let current = [];

  try {
    if (fs.existsSync(logsFile)) {
      const content = fs.readFileSync(logsFile, "utf8");
      if (content.trim().length > 0) {
        current = JSON.parse(content);
      }
    }
  } catch (err) {
    console.error("Error leyendo tamping_events.json:", err);
  }

  current.push(event);

  try {
    fs.writeFileSync(logsFile, JSON.stringify(current, null, 2), "utf8");
    console.log("Evento de tamping guardado en", logsFile);
  } catch (err) {
    console.error("Error escribiendo tamping_events.json:", err);
  }
}

// Exportar para usar en el servidor
module.exports = {
  saveTampingEvent,
};
