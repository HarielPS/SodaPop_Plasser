import styles from "./AssetsOverview.module.css";
import AssetCard from "../Diagnostics/AssetCard.jsx";

const mockAssets = [
  { name: "Machine 01", status: "Available", lastCheck: "2h ago" },
  { name: "Machine 02", status: "Busy", lastCheck: "20 min ago" },
  { name: "Track S03", status: "Critical", lastCheck: "5 min ago" },
];

export default function AssetsOverview() {
  return (
    <div className={styles.box}>
      <h3 className={styles.title}>Assets Overview</h3>

      <div className={styles.list}>
        {mockAssets.map((a, index) => (
          <AssetCard key={index} asset={a} />
        ))}
      </div>
    </div>
  );
}