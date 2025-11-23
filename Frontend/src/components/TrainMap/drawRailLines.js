// utils/drawRailLines.js
import L from "leaflet";

/**
 * Dibuja las líneas de las vías en 3 segmentos y devuelve la ruta del tren (línea 1)
 */
export function drawRailLines({
  map,
  layerGroup,
  latlngs,
  stationIcon
}) {
  const numLines = 3;
  const totalPoints = latlngs.length;
  const pointsPerLine = Math.floor(totalPoints / numLines);

  const lineColors = [
    "#ffcc00", // Línea 1
    "#00b5ff", // Línea 2
    "#7CFC00", // Línea 3
  ];

  let trainPath = null;
  let globalBounds = L.latLngBounds([]);

  for (let line = 0; line < numLines; line++) {
    const startIdx = line * pointsPerLine;
    const endIdx =
      line === numLines - 1 ? totalPoints : (line + 1) * pointsPerLine;

    const segment = latlngs.slice(startIdx, endIdx);
    if (segment.length === 0) continue;

    const color = lineColors[line];

    // Sombra gruesa
    const shaded = L.polyline(segment, {
      color,
      weight: 16,
      opacity: 0.35,
      lineCap: "round"
    }).addTo(layerGroup);

    // Línea fina encima
    L.polyline(segment, {
      color,
      weight: 4,
      opacity: 0.9,
      dashArray: "12, 8",
    }).addTo(layerGroup);

    globalBounds.extend(shaded.getBounds());

    // Estación inicio
    L.marker(segment[0], { icon: stationIcon })
      .bindPopup(`🚉 Inicio línea ${line + 1}`)
      .addTo(layerGroup);

    // Estación fin
    L.marker(segment[segment.length - 1], { icon: stationIcon })
      .bindPopup(`🚉 Fin línea ${line + 1}`)
      .addTo(layerGroup);

    // Guardar ruta del tren (línea 1)
    if (line === 0) {
      trainPath = segment;
    }
  }

  return {
    trainPath,
    bounds: globalBounds
  };
}
