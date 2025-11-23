import styles from "./AssetCard.module.css";

export default function AssetCard({ asset }) {
  return (
    <div className={styles.card}>
      <h4 className={styles.name}>{asset.name}</h4>
      <p><strong>Status:</strong> {asset.status}</p>
      <p><strong>Last check:</strong> {asset.lastCheck}</p>
    </div>
  );
}