// utils/drawRailLines.js
import L from "leaflet";

/**
 * Dibuja las líneas de las vías en 3 segmentos y devuelve:
 * - lineSegments: array con la ruta completa de cada línea
 * - bounds: límites globales para fitBounds
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

  const lineSegments = [];
  let globalBounds = L.latLngBounds([]);

  for (let line = 0; line < numLines; line++) {
    const startIdx = line * pointsPerLine;
    const endIdx =
      line === numLines - 1 ? totalPoints : (line + 1) * pointsPerLine;

    const segment = latlngs.slice(startIdx, endIdx);
    if (segment.length === 0) continue;

    const color = lineColors[line];

    // Polyline gruesa (sombreada)
    const shaded = L.polyline(segment, {
      color,
      weight: 16,
      opacity: 0.35,
      lineCap: "round"
    }).addTo(layerGroup);

    // Polyline fina encima
    L.polyline(segment, {
      color,
      weight: 4,
      opacity: 0.9,
      dashArray: "12, 8",
    }).addTo(layerGroup);

    globalBounds.extend(shaded.getBounds());

    // Marcadores inicio/fin
    L.marker(segment[0], { icon: stationIcon })
      .bindPopup(`🚉 Inicio línea ${line + 1}`)
      .addTo(layerGroup);

    L.marker(segment[segment.length - 1], { icon: stationIcon })
      .bindPopup(`🚉 Fin línea ${line + 1}`)
      .addTo(layerGroup);

    // Guardar segmento para simulación
    lineSegments[line] = segment;
  }

  return {
    lineSegments,
    bounds: globalBounds
  };
}
