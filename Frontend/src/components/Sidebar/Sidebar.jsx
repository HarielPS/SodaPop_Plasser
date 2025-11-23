import styles from "./Sidebar.module.css";
import { MdDashboard, MdSettings } from "react-icons/md";
import { FaTools } from "react-icons/fa";
import { Link } from "react-router-dom";

export default function Sidebar() {
  return (
    <aside className={styles.sidebar}>
      <h1 className={styles.logo}>Fast Tamping AI</h1>

      <nav className={styles.nav}>

        <Link className={styles.link} to="/dashboard">
          <MdDashboard className={styles.icon} />
          Dashboard
        </Link>

        <Link className={styles.link} to="/diagnostics">
          <FaTools className={styles.icon} />
          Diagnostics & History
        </Link>

        <Link className={styles.link} to="/cameras">
          <MdSettings className={styles.icon} />
          Machine Cameras
        </Link>

      </nav>
    </aside>
  );
}