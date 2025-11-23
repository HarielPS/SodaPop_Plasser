import styles from "./Header.module.css";

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.left}>
        <img src="/logoaustria.png" alt="Fast Tamping AI" className={styles.logo} />

        <div className={styles.textGroup}>
          <h1 className={styles.appName}>Fast Tamping AI</h1>
          <span className={styles.pageName}>Dashboard</span>
        </div>
      </div>

      <div className={styles.right}>
        <span className={styles.user}>Operator Online</span>
      </div>
    </header>
  );
}