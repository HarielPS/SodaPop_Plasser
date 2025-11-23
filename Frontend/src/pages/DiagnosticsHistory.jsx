import Sidebar from "../components/Sidebar/Sidebar";
import Header from "../components/Header/Header";

import AssetsOverview from "../components/Diagnostics/AssetsOverview";
import ActiveIssues from "../components/Diagnostics/ActiveIssues";
import HistoryLog from "../components/Diagnostics/HistoryLog";

import styles from "./DiagnosticsHistory.module.css";

export default function DiagnosticsHistory({ user }) {
  return (
    <div className={styles.container}>
      
      {/* Sidebar a la izquierda */}
      <Sidebar />

      {/* Contenido principal */}
      <main className={styles.main}>
        <Header user={user} />

        <h2 className={styles.title}>Diagnostics & History</h2>

        <div className={styles.grid}>
          <AssetsOverview />
          <ActiveIssues />
          <HistoryLog />
        </div>

      </main>
    </div>
  );
}