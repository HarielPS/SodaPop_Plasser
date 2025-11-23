import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import styles from "./TrainMap.module.css";

export default function TrainMap({ railwayData }) {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const routeLayerRef = useRef(null);
  const [selectedSection, setSelectedSection] = useState(null);
  const [stats, setStats] = useState({ total: 0, critical: 0, high: 0, medium: 0, low: 0 });

  useEffect(() => {
    // Crear mapa una sola vez
    if (!mapInstanceRef.current && mapRef.current) {
      mapInstanceRef.current = L.map(mapRef.current, {
        center: [45.0, 10.0], // Centro por defecto
        zoom: 5,
        zoomControl: true,
        attributionControl: false,
      });

      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        {
          maxZoom: 19,
          attribution: "",
        }
      ).addTo(mapInstanceRef.current);
    }

    // Dibujar rutas cuando cambien los datos
    if (mapInstanceRef.current && railwayData && railwayData.length > 0) {
      // Remover las rutas anteriores si existen
      if (routeLayerRef.current) {
        mapInstanceRef.current.removeLayer(routeLayerRef.current);
      }

      // Crear grupo de capas para las rutas
      routeLayerRef.current = L.layerGroup().addTo(mapInstanceRef.current);

      // Array para calcular los límites del mapa
      const allCoordinates = [];
      
      // Estadísticas
      let critical = 0, high = 0, medium = 0, low = 0;

      // Función para determinar color basado en el hundimiento
      const getColorByHundimiento = (hundimiento) => {
        if (hundimiento > 1500000) {
          critical++;
          return '#ff0000'; // Rojo - crítico
        }
        if (hundimiento > 1000000) {
          high++;
          return '#ff8800'; // Naranja - alto
        }
        if (hundimiento > 500000) {
          medium++;
          return '#ffcc00'; // Amarillo - medio
        }
        low++;
        return '#88ff00'; // Verde-amarillo - bajo
      };

      // Dibujar cada sección de la ruta
      railwayData.forEach((section) => {
        const {
          seccion_id,
          lat_inicio,
          lon_inicio,
          lat_fin,
          lon_fin,
          hundimiento_mm,
          ajuste_izquierdo_mm,
          ajuste_derecho_mm
        } = section;

        // Añadir coordenadas al array para calcular límites
        allCoordinates.push([lat_inicio, lon_inicio]);
        allCoordinates.push([lat_fin, lon_fin]);

        const lineColor = getColorByHundimiento(hundimiento_mm);

        // Crear la línea de la sección
        const routeLine = L.polyline(
          [[lat_inicio, lon_inicio], [lat_fin, lon_fin]],
          {
            color: lineColor,
            weight: 3,
            opacity: 0.8,
          }
        );

        // Popup con información de la sección
        const popupContent = `
          <div style="color: #fff; font-family: sans-serif; min-width: 200px;">
            <div style="font-weight: 600; font-size: 14px; color: ${lineColor}; margin-bottom: 8px;">
              🚂 Sección ${seccion_id}
            </div>
            <div style="font-size: 12px; line-height: 1.6;">
              <div><strong>Hundimiento:</strong> ${(hundimiento_mm / 1000).toFixed(2)} m</div>
              <div><strong>Ajuste Izq:</strong> ${ajuste_izquierdo_mm} mm</div>
              <div><strong>Ajuste Der:</strong> ${ajuste_derecho_mm} mm</div>
              <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid #444;">
                <strong>Inicio:</strong> ${lat_inicio.toFixed(4)}, ${lon_inicio.toFixed(4)}<br>
                <strong>Fin:</strong> ${lat_fin.toFixed(4)}, ${lon_fin.toFixed(4)}
              </div>
            </div>
          </div>
        `;

        routeLine.bindPopup(popupContent);
        
        // Evento de click para seleccionar sección
        routeLine.on('click', () => {
          setSelectedSection(section);
        });

        routeLine.addTo(routeLayerRef.current);

        // Añadir marcadores pequeños en los puntos si inicio y fin son diferentes
        if (lat_inicio !== lat_fin || lon_inicio !== lon_fin) {
          const pointIcon = L.divIcon({
            className: 'section-point',
            html: `<div style="
              background: ${lineColor};
              width: 6px;
              height: 6px;
              border-radius: 50%;
              border: 1px solid #fff;
              box-shadow: 0 0 4px rgba(0,0,0,0.5);
            "></div>`,
            iconSize: [6, 6],
            iconAnchor: [3, 3],
          });

          L.marker([lat_inicio, lon_inicio], { icon: pointIcon })
            .addTo(routeLayerRef.current);
        }
      });

      // Actualizar estadísticas
      setStats({
        total: railwayData.length,
        critical,
        high,
        medium,
        low
      });

      // Ajustar vista del mapa para mostrar todas las rutas
      if (allCoordinates.length > 0) {
        const bounds = L.latLngBounds(allCoordinates);
        mapInstanceRef.current.fitBounds(bounds, { padding: [20, 20] });
      }
    }
    routeLayerRef.current = L.layerGroup().addTo(mapInstanceRef.current);

    // Leer CSV
    fetch("/resources/datos_inspeccion_vias.csv")
      .then(res => res.text())
      .then(text => {
        const rows = text.trim().split("\n");
        const header = rows[0].split(",");
        const latIdx = header.indexOf("Lat_barato");
        const lonIdx = header.indexOf("Lon_barato");

        if (latIdx === -1 || lonIdx === -1) {
          console.error("No existe Lat_barato / Lon_barato en el CSV");
          return;
        }

        const latlngs = [];
        for (let i = 1; i < rows.length; i++) {
          const cols = rows[i].split(",");
          const lat = parseFloat(cols[latIdx]);
          const lon = parseFloat(cols[lonIdx]);
          if (!isNaN(lat) && !isNaN(lon)) latlngs.push([lat, lon]);
        }

        if (latlngs.length === 0) return;

        // ====== DIVIDIR EN LÍNEAS ======
        const totalPoints = latlngs.length; // 15000
        const numLines = 3;                // 3 rutas
        const pointsPerLine = Math.floor(totalPoints / numLines); // 5000

        // Colores por línea
        const lineColors = [
          "#ffcc00", // Línea 1
          "#00b5ff", // Línea 2
          "#7CFC00", // Línea 3
        ];

        let globalBounds = L.latLngBounds([]);

        // Icono de estación
        const stationIcon = L.divIcon({
          html: `<div style="
            background: #ffffff;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            border: 3px solid #ffcc00;
            box-shadow: 0 0 8px rgba(255, 204, 0, 0.9);
          "></div>`,
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        });

        // Guardar una posición para el tren (línea 1)
        let trainPosition = null;

        for (let line = 0; line < numLines; line++) {
          const startIdx = line * pointsPerLine;
          const endIdx =
            line === numLines - 1
              ? totalPoints
              : (line + 1) * pointsPerLine;

          const segment = latlngs.slice(startIdx, endIdx);
          if (segment.length === 0) continue;

          const color = lineColors[line % lineColors.length];

          // Polyline gruesa (sombreada)
          const shaded = L.polyline(segment, {
            color,
            weight: 16,
            opacity: 0.35,
            lineCap: "round",
          }).addTo(routeLayerRef.current);

          // Polyline fina encima
          L.polyline(segment, {
            color,
            weight: 4,
            opacity: 0.9,
            dashArray: "12, 8",
          }).addTo(routeLayerRef.current);

          globalBounds.extend(shaded.getBounds());

          // Marcadores inicio/fin de cada línea
          const segStart = segment[0];
          const segEnd = segment[segment.length - 1];

          L.marker(segStart, { icon: stationIcon })
            .bindPopup(`🚉 Inicio línea ${line + 1}`)
            .addTo(routeLayerRef.current);

          L.marker(segEnd, { icon: stationIcon })
            .bindPopup(`🚉 Fin línea ${line + 1}`)
            .addTo(routeLayerRef.current);

          // Tren en la línea 1 (~30% del recorrido)
          if (line === 0) {
            const idx = Math.floor(segment.length * 0.3);
            trainPosition = segment[idx];
          }
        }

        // Ajustar mapa a todo el conjunto de líneas
        if (globalBounds.isValid()) {
          mapInstanceRef.current.fitBounds(globalBounds, { padding: [40, 40] });
        }

        // ====== Tren amarillo en línea 1 ======
        if (trainPosition) {
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

          L.marker(trainPosition, { icon: trainIcon })
            .bindPopup(
              '<div style="color: #ffcc00; font-weight: 600;">🚄 Tren en tránsito (Línea 1)</div>'
            )
            .addTo(routeLayerRef.current);
        }

        // ====== (Opcional) JSON de secciones, lo dejo igual ======
        fetch("/resources/output_modelo2_sections.json")
          .then(r => r.json())
          .then(sections => {
            sections.forEach(section => {
              const p1 = [section.lat_inicio, section.lon_inicio];
              const p2 = [section.lat_fin, section.lon_fin];

              const color =
                section.hundimiento_mm > 500000
                  ? "#ff4d4d"
                  : section.hundimiento_mm > 200000
                  ? "#ffa500"
                  : "#00ccff";

              L.polyline([p1, p2], {
                color,
                weight: 18,
                opacity: 0.3,
                lineCap: "round",
              })
                .bindPopup(`
                  <b>Sección ${section.seccion_id}</b><br/>
                  Hundimiento: ${section.hundimiento_mm.toFixed(2)} mm
                `)
                .addTo(routeLayerRef.current);
            });
          })
          .catch(err =>
            console.error("Error cargando JSON de secciones", err)
          );
      })
      .catch(err => console.error("Error cargando CSV", err));

    return () => {
      // Cleanup de capas, no del mapa completo
    };
  }, [railwayData]);

  // Cleanup al desmontar el componente
  useEffect(() => {
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  return (
    <div className={styles.mapContainer}>
      <div className={styles.header}>
        <h3 className={styles.title}>Railway Map</h3>
        <div className={styles.sectionCount}>
          {stats.total} Secciones
        </div>
      </div>
      
      <div ref={mapRef} className={styles.map}></div>
      
      <div className={styles.legend}>
        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: '#ff0000' }}></span>
          <span className={styles.legendLabel}>Crítico ({stats.critical})</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: '#ff8800' }}></span>
          <span className={styles.legendLabel}>Alto ({stats.high})</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: '#ffcc00' }}></span>
          <span className={styles.legendLabel}>Medio ({stats.medium})</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendColor} style={{ background: '#88ff00' }}></span>
          <span className={styles.legendLabel}>Bajo ({stats.low})</span>
        </div>
      </div>

      {selectedSection && (
        <div className={styles.selectedInfo}>
          <div className={styles.selectedHeader}>
            <span>Sección {selectedSection.seccion_id} seleccionada</span>
            <button 
              className={styles.closeBtn}
              onClick={() => setSelectedSection(null)}
            >
              ✕
            </button>
          </div>
          <div className={styles.selectedData}>
            <div>Hundimiento: {(selectedSection.hundimiento_mm / 1000).toFixed(2)} m</div>
            <div>Ajustes: {selectedSection.ajuste_izquierdo_mm} mm / {selectedSection.ajuste_derecho_mm} mm</div>
          </div>
        </div>
      )}
    </div>
  );
}
