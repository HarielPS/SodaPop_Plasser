import styles from "./Sidebar.module.css";
import { MdDashboard, MdSettings } from "react-icons/md";
import { FaTools } from "react-icons/fa";

export default function Sidebar() {
  return (
    <aside className={styles.sidebar}>
      <h1 className={styles.logo}>Fast Tamping AI</h1>

      <nav className={styles.nav}>
        <a className={styles.link} href="#">
          <MdDashboard className={styles.icon} />
          Dashboard
        </a>

        <a className={styles.link} href="#">
          <FaTools className={styles.icon} />
          Lift Diagnosing
        </a>

        <a className={styles.link} href="#">
          <MdSettings className={styles.icon} />
          Track Parameters
        </a>
      </nav>
    </aside>
  );
}