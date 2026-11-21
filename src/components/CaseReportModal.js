
import React from 'react';

export default function CaseReportModal({ report, onClose }) {
    return (
        <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            width: '100vw',
            height: '100vh',
            backgroundColor: 'rgba(0,0,0,0.85)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 9999
        }}>
            <div style={{
                backgroundColor: '#111b21',
                border: '1px solid #dc2626',
                borderRadius: '8px',
                padding: '1.5rem',
                maxWidth: '550px',
                width: '90%',
                color: '#e9edef'
            }}>
                <h2 style={{ color: '#ef4444', marginTop: 0 }}>🚨 FORENSIC CASE REPORT GENERATED</h2>
                <p style={{ fontSize: '0.85rem', color: '#8696a0' }}>Threshold Limit Exceeded (10 Warnings Reached)</p>

                <div style={{ backgroundColor: '#202c33', padding: '1rem', borderRadius: '6px', fontSize: '0.9rem' }}>
                    <p><strong><b>Case ID:</b></strong> {report.case_id}</p>
                    <p><strong><b>Offender Mobile:</b></strong> {report.offender_mobile_number}</p>
                    <p><strong><b>Timestamp (IST):</b></strong> {report.timestamp}</p>
                    <p><strong><b>Toxicity Risk:</b></strong> {report.toxicity_risk_percentage}</p>
                    <p><strong><b>SHA-256 Hash:</b></strong> <code style={{ color: '#38bdf8', fontSize: '0.75rem' }}>{report.sha256_chain_hash}</code></p>

                    <h4 style={{ color: '#f87171', marginBottom: '0.4rem' }}>Top 3 LIME Explanation Drivers:</h4>
                    <ul>
                        {report.top_3_lime_explanations.map((item, idx) => (
                            <li key={idx}>
                                Word: <strong>'{item.word}'</strong> | Attribution: {item.attribution.toFixed(4)}
                            </li>
                        ))}
                    </ul>

                    <p style={{ fontSize: '0.75rem', color: '#94a3b8', fontStyle: 'italic', marginTop: '0.75rem' }}>
                        🔒 {report.legal_notice}
                    </p>
                </div>

                <button
                    onClick={onClose}
                    style={{ width: '100%', marginTop: '1rem', padding: '0.6rem', backgroundColor: '#dc2626', color: '#fff', border: 'none', borderRadius: '4px', fontWeight: 'bold', cursor: 'pointer' }}>
                    Close Case Report
                </button>
            </div>
        </div>
    );
}