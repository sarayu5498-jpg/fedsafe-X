import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = "http://localhost:8000/api";

export default function AdminDashboard({ onClose }) {
    const [stats, setStats] = useState(null);
    const [reports, setReports] = useState([]);

    useEffect(() => {
        const fetchAdminData = async () => {
            try {
                const statsRes = await axios.get(`${API_BASE}/admin/dashboard_stats`);
                setStats(statsRes.data);

                const reportsRes = await axios.get(`${API_BASE}/admin/reports`);
                setReports(reportsRes.data.reports);
            } catch (err) {
                console.error("Admin Dashboard Error:", err);
            }
        };

        fetchAdminData();
        const interval = setInterval(fetchAdminData, 3000);
        return () => clearInterval(interval);
    }, []);

    if (!stats) {
        return (
            <div style={{ padding: '2rem', textAlign: 'center', color: '#38bdf8' }}>
                Loading Admin Real-Time Analytics...
            </div>
        );
    }

    return (
        <div style={{
            backgroundColor: '#1e293b',
            borderRadius: '12px',
            border: '1px solid #334155',
            padding: '1.5rem',
            color: '#f8fafc',
            maxHeight: '85vh',
            overflowY: 'auto'
        }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '1px solid #334155', paddingBottom: '0.75rem' }}>
                <div>
                    <h2 style={{ margin: 0, color: '#38bdf8', fontSize: '1.2rem', fontWeight: '700' }}>
                        📊 Federated Safety Audit Control Panel
                    </h2>
                </div>
                <button
                    onClick={onClose}
                    style={{ backgroundColor: '#334155', color: '#f8fafc', border: 'none', borderRadius: '6px', padding: '0.4rem 0.8rem', cursor: 'pointer', fontWeight: '600' }}>
                    ✕
                </button>
            </div>

            {/* Metric Cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                <div style={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '1rem', textAlign: 'center' }}>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block' }}>TOTAL ACCOUNTS</span>
                    <span style={{ fontSize: '1.5rem', fontWeight: '700', color: '#38bdf8' }}>{stats.total_users}</span>
                </div>
                <div style={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '1rem', textAlign: 'center' }}>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block' }}>ACTIVE ACCOUNTS</span>
                    <span style={{ fontSize: '1.5rem', fontWeight: '700', color: '#34d399' }}>{stats.active_users}</span>
                </div>
                <div style={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '1rem', textAlign: 'center' }}>
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block' }}>BLOCKED ACCOUNTS</span>
                    <span style={{ fontSize: '1.5rem', fontWeight: '700', color: '#f87171' }}>{stats.blocked_users}</span>
                </div>
            </div>

            {/* PDF Reports List */}
            <h3 style={{ fontSize: '0.95rem', color: '#38bdf8', marginBottom: '0.6rem' }}>📄 Generated Forensic Reports</h3>
            <div style={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '1rem', marginBottom: '1.5rem' }}>
                {reports.length === 0 ? (
                    <p style={{ fontSize: '0.8rem', color: '#94a3b8', margin: 0 }}>No forensic PDF reports found.</p>
                ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {reports.map((rep) => (
                            <div key={rep} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#1e293b', padding: '0.5rem 0.8rem', borderRadius: '6px' }}>
                                <span style={{ fontSize: '0.8rem', color: '#f8fafc' }}>{rep}</span>
                                <a
                                    href={`${API_BASE}/admin/download_report/${rep}`}
                                    target="_blank"
                                    rel="noreferrer"
                                    style={{ backgroundColor: '#0284c7', color: '#fff', padding: '0.25rem 0.6rem', borderRadius: '4px', textDecoration: 'none', fontWeight: 'bold', fontSize: '0.75rem' }}>
                                    Download Report
                                </a>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Excel Warning Ledger View */}
            <h3 style={{ fontSize: '0.95rem', color: '#34d399', marginBottom: '0.6rem' }}>📊 User Warning Ledger Records</h3>
            <div style={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px', padding: '1rem' }}>
                {stats.ledger_records.length === 0 ? (
                    <p style={{ fontSize: '0.8rem', color: '#94a3b8', margin: 0 }}>No user warning records found.</p>
                ) : (
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid #334155', color: '#94a3b8' }}>
                                <th style={{ padding: '0.5rem' }}>Username</th>
                                <th style={{ padding: '0.5rem' }}>Mobile Number</th>
                                <th style={{ padding: '0.5rem' }}>Email ID</th>
                                <th style={{ padding: '0.5rem' }}>Warnings</th>
                                <th style={{ padding: '0.5rem' }}>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {stats.ledger_records.map((r, i) => (
                                <tr key={i} style={{ borderBottom: '1px solid #1e293b' }}>
                                    <td style={{ padding: '0.5rem' }}>{r["Username"]}</td>
                                    <td style={{ padding: '0.5rem' }}>{r["Mobile Number"]}</td>
                                    <td style={{ padding: '0.5rem' }}>{r["Email ID"]}</td>
                                    <td style={{ padding: '0.5rem', fontWeight: 'bold', color: r["Warning Count"] >= 10 ? '#ef4444' : '#f59e0b' }}>
                                        {r["Warning Count"]}/10
                                    </td>
                                    <td style={{ padding: '0.5rem', fontWeight: 'bold', color: r["Account Status"] === 'BLOCKED' ? '#ef4444' : '#34d399' }}>
                                        {r["Account Status"]}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
}