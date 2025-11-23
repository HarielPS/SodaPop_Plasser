import Sidebar from "../components/Sidebar/Sidebar";
import Header from "../components/Header/Header";
import styles from "./MachineCamara.module.css";


export default function TrainCameras({ user }) {
  return (
    <div className={styles.container}>

      {/* Sidebar izquierda */}
      <Sidebar />

      {/* Contenido principal */}
      <main className={styles.main}>
        <Header user={user} />

        <h2 className={styles.title}>Train Cameras Monitoring</h2>

        <div className={styles.grid}>
          
          {/* Cámara 1 */}
          <div className={styles.cameraCard}>
            <h3 className={styles.camTitle}>Front Camera</h3>
            <video 
              className={styles.video}
              controls
              autoPlay
              loop
              muted
              src="/videos/Camara/video1.mp4"
            />
          </div>

          {/* Cámara 2 */}
          <div className={styles.cameraCard}>
            <h3 className={styles.camTitle}>Cabin Camera</h3>
            <video 
              className={styles.video}
              controls
              autoPlay
              loop
              muted
              src="/videos/Camara/video2.mp4"
            />
          </div>

          {/* Cámara 3 */}
          <div className={styles.cameraCard}>
            <h3 className={styles.camTitle}>Rear Camera</h3>
            <video 
              className={styles.video}
              controls
              autoPlay
              loop
              muted
              src="/videos/Camara/video3.mp4"
            />
          </div>

        </div>

      </main>
    </div>
  );
}
