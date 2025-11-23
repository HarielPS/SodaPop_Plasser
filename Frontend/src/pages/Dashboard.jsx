import Sidebar from "../components/Sidebar/Sidebar.jsx";
import Header from "../components/Header/Header.jsx";
import TrainMap from "../components/TrainMap/TrainMap.jsx";
import AlertsPanel from "../components/AlertsPanel/XAlertsPanel.jsx";
import MachinesStatus from "../components/MachinesStatus/MachinesStatus.jsx";
import ArrivalTimer from "../components/ArrivalTimer/ArrivalTimer.jsx";
import KPIsPanel from "../components/KPIsPanel/KPIsPanel.jsx";

import styles from "./Dashboard.module.css";

export default function Dashboard() {
  return (
    <div className={styles.layout}>
      <Sidebar />

      <div className={styles.content}>
        <Header />

        <div className={styles.grid}>
          <TrainMap />
          <AlertsPanel />
          <ArrivalTimer />
          <MachinesStatus />
          <KPIsPanel />
        </div>
      </div>
    </div>
  );
}