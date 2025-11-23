import styles from "./Header.module.css";

export default function Header({ user }) {
  return (
    <header className={styles.header}>
      <h2 className={styles.title}>Fast Tamping AI</h2>

      <div className={styles.userBox}>
        <span className={styles.role}>Operator</span>
        <span className={styles.name}>
          {user?.name} {user?.lastname}
        </span>
      </div>
    </header>
  );
}