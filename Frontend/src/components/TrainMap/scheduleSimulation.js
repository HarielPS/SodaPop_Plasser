// utils/scheduleSimulation.js
import L from "leaflet";

/** Encuentra el punto más cercano */
function findClosestIndex(path, targetLat, targetLon) {
  let best = 0, bestDist = Infinity;
  for (let i = 0; i < path.length; i++) {
    const [lat, lon] = path[i];
    const d = (lat - targetLat) ** 2 + (lon - targetLon) ** 2;
    if (d < bestDist) { bestDist = d; best = i; }
  }
  return best;
}

/** Animación sin duplicar markers */
function animateMarkerAlongPath({
  path,
  durationMs,
  icon,
  marker,           // 🔵 ahora recibe marker siempre
  registry,
  isTrain = false,
  lineIdx,
  blockState,
  onStep,
  onComplete,
}) {
  if (!marker || !path || path.length < 2) return;

  const steps = path.length - 1;
  const stepTime = durationMs / steps;
  let idx = 0;

  const safeDistance = 10;

  const intervalId = setInterval(() => {

    // 🚧 Tren detenido por bloqueo
    if (isTrain && blockState.blocks[lineIdx]) {
      const blocked = blockState.blocks[lineIdx].idx;
      if (idx + safeDistance >= blocked) return;
    }

    idx++;
    if (idx >= path.length) {
      clearInterval(intervalId);
      if (onComplete) onComplete(marker);
      return;
    }

    marker.setLatLng(path[idx]);
    if (onStep) onStep(idx, marker);

  }, stepTime);

  registry.push({ marker, intervalId });
}

/** PROCESO PRINCIPAL */
export function scheduleSimulation({
  map,
  layerGroup,
  lineSegments,
  schedule,
  registryRef,
}) {
  const registry = [];
  registryRef.current = registry;

  const blueMachines = {}; // 🔵 UNA sola máquina por ID
  const blockState = { blocks: {} };
  const tampingLaunchedForLine = {};

  const trainIcon = L.divIcon({
    html: `<div style="background:#ffcc00;width:24px;height:16px;border-radius:4px;
    box-shadow:0 0 15px #fc0;display:flex;align-items:center;justify-content:center">🚄</div>`,
    iconSize: [24, 16],
    iconAnchor: [12, 8],
    options: { type: "train" },
  });

  const tampingIcon = L.divIcon({
    html: `<div style="background:#00e0ff;width:22px;height:18px;border-radius:4px;
    box-shadow:0 0 12px #0ef;display:flex;align-items:center;justify-content:center">🛠️</div>`,
    iconSize: [22, 18],
    iconAnchor: [11, 9],
    options: { type: "tamping" },
  });

  function triggerTampingForLine(lineIdx) {
    const fullPath = lineSegments[lineIdx];
    if (!fullPath) return;

    const machines = schedule.tampingMachines.filter(m => (m.line - 1) === lineIdx);

    machines.forEach((tm) => {
      let targetIndex =
        tm.targetIndex ??
        findClosestIndex(fullPath, tm.targetLat, tm.targetLon);

      const [zLat, zLon] = fullPath[targetIndex];

      // 🚨 bloquear línea desde YA
      blockState.blocks[lineIdx] = { idx: 0 };

      // 🔵 dibujar zona
      const zoneMarker = L.circleMarker([zLat, zLon], {
        radius: 8, fillColor: "#00e0ff", fillOpacity: 0.4,
        color: "#00e0ff", weight: 2
      }).addTo(layerGroup);

      registry.push({ marker: zoneMarker, intervalId: null });

      // popup
      L.popup().setLatLng([zLat, zLon]).setContent(`
        <b>⚠ Hundimiento detectado</b><br>
        Línea ${tm.line}<br>${tm.description || ""}
      `).openOn(map);

      // rutas
      const pathToTarget = fullPath.slice(0, targetIndex + 1);
      const restPath = fullPath.slice(targetIndex);

      const dwellMs = 4000;
      const travelMs = 3000;

      // 🛠️ Crear marker azul SOLO UNA VEZ
      if (!blueMachines[tm.id]) {
        blueMachines[tm.id] = L.marker(pathToTarget[0], { icon: tampingIcon })
          .addTo(layerGroup);
      }
      const blue = blueMachines[tm.id];

      const onStepTo = (i) => blockState.blocks[lineIdx] = { idx: i };
      const onStepRest = (i) => blockState.blocks[lineIdx] = { idx: targetIndex + i };

      // IR a la zona
        animateMarkerAlongPath({
        path: pathToTarget,
        durationMs: travelMs,
        icon: tampingIcon,
        marker: blue,
        registry,
        isTrain: false,
        lineIdx,
        blockState,
        onStep: onStepTo,

        onComplete: () => {

            // Esperar 4 s en la zona de reparación
            setTimeout(() => {

            // 🟦 1) QUITAR el círculo de reparación ANTES de avanzar
            if (zoneMarker) {
                layerGroup.removeLayer(zoneMarker);
            }

            // 🟦 2) LIBERAR la línea inmediatamente (para que trenes puedan salir)
            blockState.blocks[lineIdx] = null;

            // 🟦 3) AVANZAR hacia adelante por el resto de la línea
            animateMarkerAlongPath({
                path: restPath,
                durationMs: 3000, // velocidad fija 3 s
                icon: tampingIcon,
                marker: blue,
                registry,
                isTrain: false,
                lineIdx,
                blockState,

                onStep: (i) => {
                // mantener actualizado el índice de bloqueo mientras se mueve
                blockState.blocks[lineIdx] = { idx: targetIndex + i };
                },

                onComplete: () => {
                // ya no hay nada más que hacer
                // la línea ya fue liberada arriba
                },
            });

            }, dwellMs); // dwellMs = 4000 ms (4 s)
        },
        });


    });
  }

  /** TRENES */
  schedule.trains.forEach((train) => {
    const lineIdx = train.line - 1;
    const path = lineSegments[lineIdx];
    const departMs = train.departureSec * 1000;
    const travelMs = train.travelTimeSec * 1000;

    const launch = () => {

      if (blockState.blocks[lineIdx]) {
        setTimeout(launch, 1000);
        return;
      }

      const marker = L.marker(path[0], { icon: trainIcon }).addTo(layerGroup);

      animateMarkerAlongPath({
        path,
        durationMs: travelMs,
        icon: trainIcon,
        marker,
        registry,
        isTrain: true,
        lineIdx,
        blockState,
        onComplete: () => {

          if (!tampingLaunchedForLine[lineIdx]) {
            tampingLaunchedForLine[lineIdx] = true;
            triggerTampingForLine(lineIdx);
          }

          setTimeout(launch, departMs);
        },
      });

    };

    setTimeout(launch, departMs);
  });

}

/** Limpieza */
export function clearSimulation(registryRef, layerGroup) {
  registryRef.current.forEach(item => {
    if (item.intervalId) clearInterval(item.intervalId);
    if (item.marker) layerGroup.removeLayer(item.marker);
  });
  registryRef.current = [];
}
