export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="loading">
      <span className="loading-bars" aria-hidden="true"><i /><i /><i /></span>
      <span>{label}</span>
    </div>
  );
}
