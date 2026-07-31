const STATUS_STYLES = {
  VERIFIED: { label: "VERIFIED", className: "status-verified" },
  FLAGGED: { label: "FLAGGED FOR REVIEW", className: "status-flagged" },
  FAILED: { label: "FAILED", className: "status-failed" },
  UNKNOWN: { label: "UNKNOWN", className: "status-unknown" },
};

const FIELD_LABELS = {
  name: "Name",
  date_of_birth: "Date of birth",
  document_number: "Document number",
};

export default function VerificationReport({ result }) {
  const status = STATUS_STYLES[result.status] ?? STATUS_STYLES.UNKNOWN;

  return (
    <div className="report-card">
      <div className={`status-banner ${status.className}`}>{status.label}</div>
      <p className="report-filename">{result.filename}</p>

      <section className="report-section">
        <h3>Document forgery detection</h3>
        {result.forgery.error ? (
          <p className="field-reason">{result.forgery.error}</p>
        ) : (
          <>
            <p>
              <strong>{result.forgery.label.toUpperCase()}</strong>{" "}
              <span className="confidence">
                ({(result.forgery.confidence * 100).toFixed(1)}% confidence)
              </span>
            </p>
            {result.forgery.regions && result.forgery.regions.length > 0 && (
              <table className="fields-table">
                <tbody>
                  {result.forgery.regions.map((region) => (
                    <tr key={region.category}>
                      <td>{region.category.replaceAll("_", " ")}</td>
                      <td className={region.label === "tampered" ? "invalid" : "valid"}>
                        {region.label}
                      </td>
                      <td className="field-reason">{(region.confidence * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </section>

      {result.ocr && (
        <section className="report-section">
          <h3>Extracted fields</h3>
          <table className="fields-table">
            <tbody>
              {Object.entries(FIELD_LABELS).map(([key, label]) => {
                const field = result.ocr[key];
                return (
                  <tr key={key}>
                    <td>{label}</td>
                    <td>{field?.raw || <em>not found</em>}</td>
                    <td className={field?.valid ? "valid" : "invalid"}>
                      {field?.valid ? "✓" : "✗"}
                    </td>
                    <td className="field-reason">{field?.reason ?? ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
