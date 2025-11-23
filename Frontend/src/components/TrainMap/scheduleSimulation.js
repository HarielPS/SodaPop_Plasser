// utils/scheduleSimulation.js
import L from "leaflet";

/**
 * Crea un marker que se mueve a lo largo de un path en durationMs
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

  // Guardar para poder limpiar después
  registry.push({ marker, intervalId });
}

/**
 * Programa trenes normales y máquinas de tamping
 * según el schedule del JSON.
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

  // Iconos
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

  // ========= TRENES NORMALES =========
  (schedule.trains || []).forEach((train) => {
    const lineIdx = (train.line || 1) - 1;
    const path = lineSegments[lineIdx];
    if (!path) return;

    const departMs = (train.departureSec || 0) * 1000;
    const durationMs = (train.travelTimeSec || 180) * 1000;

    setTimeout(() => {
      animateMarkerAlongPath({
        layerGroup,
        path,
        durationMs,
        icon: trainIcon,
        registry,
      });
    }, departMs);
  });

  // ========= MÁQUINAS DE TAMPING =========
  (schedule.tampingMachines || []).forEach((tm) => {
    const lineIdx = (tm.line || 1) - 1;
    const fullPath = lineSegments[lineIdx];
    if (!fullPath) return;

    const departMs = (tm.departureSec || 0) * 1000;
    const travelMs = (tm.travelTimeToTargetSec || 120) * 1000;
    const dwellMs = (tm.dwellTimeSec || 60) * 1000;
    const targetIndex = Math.min(
      tm.targetIndex || Math.floor(fullPath.length / 2),
      fullPath.length - 1
    );

    const pathToTarget = fullPath.slice(0, targetIndex + 1);
    const pathBack = pathToTarget.slice().reverse();

    setTimeout(() => {
      // Ir a la zona sombreada
      animateMarkerAlongPath({
        layerGroup,
        path: pathToTarget,
        durationMs: travelMs,
        icon: tampingIcon,
        registry,
        onComplete: (marker) => {
          // Esperar en la zona (trabajando)
          setTimeout(() => {
            if (tm.returnToBase) {
              // Regresar
              animateMarkerAlongPath({
                layerGroup,
                path: pathBack,
                durationMs: travelMs,
                icon: tampingIcon,
                registry,
                onComplete: () => {
                  // Quitar marker al regresar
                  layerGroup.removeLayer(marker);
                },
              });
            } else {
              // Seguir avanzando hacia adelante
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
    }, departMs);
  });
}

/**
 * Limpia todos los markers y animaciones activos
 */
export function clearSimulation(registryRef, layerGroup) {
  const registry = registryRef.current || [];
  registry.forEach(({ marker, intervalId }) => {
    clearInterval(intervalId);
    if (layerGroup && marker) layerGroup.removeLayer(marker);
  });
  registryRef.current = [];
}
