<<<<<<< HEAD
import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
=======
import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
>>>>>>> 92aa4c08ea8ec3fd8cc12acb59d56b2a1c5b9086
import styles from "./TrainMap.module.css";

// Lógica externa
import { animateTrainOnPath } from "./animateTrain";
import { drawRailLines } from "./drawRailLines";

export default function TrainMap() {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const routeLayerRef = useRef(null);

  const trainMarkerRef = useRef(null);
  const trainAnimationRef = useRef(null);

  useEffect(() => {
    // Crear mapa
    if (!mapInstanceRef.current && mapRef.current) {
      mapInstanceRef.current = L.map(mapRef.current, {
        center: [45.0, 10.0],
        zoom: 5,
        zoomControl: true,
      });

      L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        { maxZoom: 19 }
      ).addTo(mapInstanceRef.current);
    }

    // Crear capa
    if (routeLayerRef.current) {
      mapInstanceRef.current.removeLayer(routeLayerRef.current);
    }
    routeLayerRef.current = L.layerGroup().addTo(mapInstanceRef.current);

    // ============ LEER CSV ============
    fetch("/resources/datos_inspeccion_vias.csv")
      .then((res) => res.text())
      .then((text) => {
        const rows = text.trim().split("\n");
        const header = rows[0].split(",");
<<<<<<< HEAD
=======

        const latIdx = header.indexOf("Lat_barato");
>>>>>>> 92aa4c08ea8ec3fd8cc12acb59d56b2a1c5b9086
        const lonIdx = header.indexOf("Lon_barato");
        const latIdx = header.indexOf("Lat_barato");

<<<<<<< HEAD
        if (latIdx === -1 || lonIdx === -1) {
          console.error("No existe Lat_caro / Lon_caro en el CSV");
          return;
        }

=======
>>>>>>> 92aa4c08ea8ec3fd8cc12acb59d56b2a1c5b9086
        const latlngs = [];
        for (let i = 1; i < rows.length; i++) {
          const cols = rows[i].split(",");
          const lat = parseFloat(cols[latIdx]);
          const lon = parseFloat(cols[lonIdx]);
          if (!isNaN(lat) && !isNaN(lon)) latlngs.push([lat, lon]);
        }

        if (latlngs.length === 0) return;

        // Icono de estación
        const stationIcon = L.divIcon({
          html: `<div style="
            background:white;
            width:10px;
            height:10px;
            border-radius:50%;
            border:3px solid #ffcc00;
            box-shadow:0 0 8px rgba(255,204,0,0.8);
          "></div>`,
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        });

        // ======= 🎨 DIBUJAR LÍNEAS ESTILIZADAS =======
        const { trainPath, bounds } = drawRailLines({
          map: mapInstanceRef.current,
          layerGroup: routeLayerRef.current,
          latlngs,
          stationIcon
        });

        // ======= 🚄 ANIMAR TREN =======
        animateTrainOnPath({
          map: mapInstanceRef.current,
          layerGroup: routeLayerRef.current,
          path: trainPath,
          totalDuration: 20000,
          trainMarkerRef,
          trainAnimationRef,
        });

        // Ajustar vista
        if (bounds.isValid()) {
          mapInstanceRef.current.fitBounds(bounds, { padding: [40, 40] });
        }
<<<<<<< HEAD

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
        fetch("/resources/output_lifts.json")
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
=======
      });
>>>>>>> 92aa4c08ea8ec3fd8cc12acb59d56b2a1c5b9086

    return () => {
      if (trainAnimationRef.current) clearInterval(trainAnimationRef.current);
    };
  }, []);

  return <div ref={mapRef} className={styles.map}></div>;
}
