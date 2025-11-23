import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
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

        const latIdx = header.indexOf("Lat_barato");
        const lonIdx = header.indexOf("Lon_barato");

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
      });

    return () => {
      if (trainAnimationRef.current) clearInterval(trainAnimationRef.current);
    };
  }, []);

  return <div ref={mapRef} className={styles.map}></div>;
}
