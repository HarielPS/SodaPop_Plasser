// utils/scheduleSimulation.js
import L from "leaflet";

/**
 * Encuentra el índice del punto de la ruta más cercano a (lat, lon)
 */
function findClosestIndex(path, targetLat, targetLon) {
  if (!path || path.length === 0) return 0;

  let bestIdx = 0;
  let bestDist = Infinity;

  for (let i = 0; i < path.length; i++) {
    const [lat, lon] = path[i];
    const dLat = lat - targetLat;
    const dLon = lon - targetLon;
    const distSq = dLat * dLat + dLon * dLon;
    if (distSq < bestDist) {
      bestDist = distSq;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * Mueve un marker a lo largo de un path en durationMs
 */
function animateMarkerAlongPath({
  layerGroup,
  path,
  durationMs,
  icon,
  onComplete,
  registry
}) {
  if (!path || path.length < 2) return;

  const marker = L.marker(path[0], { icon }).addTo(layerGroup);
  const steps = path.length - 1;
  const stepTime = durationMs / steps;

  let idx = 0;
  const intervalId = setInterval(() => {
    idx++;
    if (idx >= path.length) {
      clearInterval(intervalId);
      if (onComplete) onComplete(marker);
      return;
    }
    marker.setLatLng(path[idx]);
  }, stepTime);

  // Guardamos el intervalo y el marker para poder limpiarlos luego
  registry.push({ marker, intervalId });
}

/**
 * Programa trenes normales (en bucle) y máquinas de tamping.
 */
export function scheduleSimulation({
  map,
  layerGroup,
  lineSegments,
  schedule,
  registryRef
}) {
  if (!schedule || !lineSegments) return;

  const registry = [];
  registryRef.current = registry;

  // Icono de tren
  const trainIcon = L.divIcon({
    html: `<div style="
      background: #ffcc00;
      width: 24px;
      height: 16px;
      border-radius: 4px;
      box-shadow: 0 0 15px rgba(255, 204, 0, 0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
    ">🚄</div>`,
    iconSize: [24, 16],
    iconAnchor: [12, 8],
  });

  // Icono de máquina de tamping
  const tampingIcon = L.divIcon({
    html: `<div style="
      background: #00e0ff;
      width: 22px;
      height: 18px;
      border-radius: 4px;
      box-shadow: 0 0 12px rgba(0, 224, 255, 0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
    ">🛠️</div>`,
    iconSize: [22, 18],
    iconAnchor: [11, 9],
  });

  // Icono / estilo de la zona de tamping
  const zoneStyle = {
    radius: 8,
    color: "#00e0ff",
    weight: 2,
    fillColor: "#00e0ff",
    fillOpacity: 0.3
  };

  // ============================
  //         TRENES (BUCLE)
  // ============================
  (schedule.trains || []).forEach((train) => {
    const lineIdx = (train.line || 1) - 1;
    const path = lineSegments[lineIdx];
    if (!path) return;

    const departMs = (train.departureSec || 0) * 1000;
    const travelMs = (train.travelTimeSec || 180) * 1000;

    // Lanza un tren una sola vez
    const launchSingleTrain = () => {
      animateMarkerAlongPath({
        layerGroup,
        path,
        durationMs: travelMs,
        icon: trainIcon,
        registry,
        onComplete: () => {
          // Cuando llega al final, vuelve a programarse tras el mismo offset
          setTimeout(launchSingleTrain, departMs);
        },
      });
    };

    // Primer lanzamiento
    setTimeout(launchSingleTrain, departMs);
  });

  // ============================
  //   MÁQUINAS DE TAMPING
  // ============================
  (schedule.tampingMachines || []).forEach((tm) => {
    const lineIdx = (tm.line || 1) - 1;
    const fullPath = lineSegments[lineIdx];
    if (!fullPath) return;

    const departMs = (tm.departureSec || 0) * 1000;
    const travelMs = (tm.travelTimeToTargetSec || 120) * 1000;
    const dwellMs = (tm.dwellTimeSec || 60) * 1000;

    // ======================
    // 1) Elegir índice objetivo
    // ======================
    let targetIndex;

    // a) Si viene targetLat/targetLon en el JSON -> buscar punto más cercano en la ruta
    if (typeof tm.targetLat === "number" && typeof tm.targetLon === "number") {
      targetIndex = findClosestIndex(fullPath, tm.targetLat, tm.targetLon);
    }
    // b) Si viene targetIndex, usarlo (clamp al rango)
    else if (typeof tm.targetIndex === "number") {
      targetIndex = Math.min(
        Math.max(tm.targetIndex, 0),
        fullPath.length - 1
      );
    }
    // c) Si no hay nada, usar la mitad de la ruta
    else {
      targetIndex = Math.floor(fullPath.length / 2);
    }

    const targetLatLng = fullPath[targetIndex];

    // ======================
    // 2) Dibujar la zona de tamping en la vía (punto visible)
    // ======================
    if (targetLatLng) {
      const [zLat, zLon] = targetLatLng;
      const zoneMarker = L.circleMarker([zLat, zLon], zoneStyle)
        .bindPopup(`<b>${tm.id}</b><br/>Zona de tamping`)
        .addTo(layerGroup);

      // Lo registramos para poder limpiarlo después
      registry.push({ marker: zoneMarker, intervalId: null });
    }

    const pathToTarget = fullPath.slice(0, targetIndex + 1);
    const pathBack = pathToTarget.slice().reverse();

    // ======================
    // 3) Misión de tamping: ir → esperar → regresar o avanzar
    // ======================
    const launchTampingMission = () => {
      // Ir hasta la zona de reparación
      animateMarkerAlongPath({
        layerGroup,
        path: pathToTarget,
        durationMs: travelMs,
        icon: tampingIcon,
        registry,
        onComplete: (marker) => {
          // Esperar trabajando en la zona
          setTimeout(() => {
            if (tm.returnToBase) {
              // Regresa al origen
              animateMarkerAlongPath({
                layerGroup,
                path: pathBack,
                durationMs: travelMs,
                icon: tampingIcon,
                registry,
                onComplete: () => {
                  layerGroup.removeLayer(marker);
                },
              });
            } else {
              // Sigue avanzando hacia adelante desde la zona
              const restPath = fullPath.slice(targetIndex);
              animateMarkerAlongPath({
                layerGroup,
                path: restPath,
                durationMs: travelMs,
                icon: tampingIcon,
                registry,
              });
            }
          }, dwellMs);
        },
      });
    };

    // Por ahora, una misión única (no en bucle) para no saturar la línea.
    setTimeout(launchTampingMission, departMs);
  });
}

/**
 * Limpia todos los markers y animaciones activos
 */
export function clearSimulation(registryRef, layerGroup) {
  const registry = registryRef.current || [];
  registry.forEach(({ marker, intervalId }) => {
    if (intervalId) clearInterval(intervalId);
    if (layerGroup && marker) layerGroup.removeLayer(marker);
  });
  registryRef.current = [];
}
