import { BarChart3, CalendarClock, Compass, Sparkles, Star } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { EmptyState } from "../components/EmptyState";
import { Loading } from "../components/Loading";
import { getAnalytics } from "../lib/api";
import { useAsync } from "../lib/useAsync";

const colors = ["#0f766e", "#f59e0b", "#be123c", "#2563eb", "#7c3aed", "#475569", "#16a34a"];

function topEntries(record: Record<string, number>, limit = 8) {
  return Object.entries(record)
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([name, value]) => ({ name, value }));
}


export function AnalyticsPage({ token }: { token: string }) {
  const { data, loading, error } = useAsync(() => getAnalytics(token), [token]);

  if (loading) return <Loading label="Building taste profile" />;
  if (error) return <EmptyState icon={BarChart3} title="Could not load analytics" body={error} />;
  if (!data) return null;
  if (data.movies_watched === 0) {
    return <EmptyState icon={BarChart3} title="Taste profile is empty" body="Watch and rate a few movies to unlock analytics." />;
  }

  const genres = topEntries(data.genre_breakdown);
  const ratings = topEntries(data.rating_distribution, 10);
  const decades = topEntries(data.decade_breakdown);
  const clusters = topEntries(data.cluster_breakdown);
  const favoriteGenre = genres[0]?.name ?? "still forming";
  const signatureDecade = decades[0]?.name ?? "mixed eras";
  const topDiscovery = clusters[0]?.name ?? "Genre explorer";
  const ratingMood = data.avg_rating && data.avg_rating >= 4 ? "selective superfan" : data.avg_rating && data.avg_rating >= 3 ? "curious critic" : "bold explorer";

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <p className="eyebrow">Taste intelligence</p>
          <h1>Taste Lab</h1>
          <p>A livelier readout of the genres, eras, and rating habits shaping your recommendations.</p>
        </div>
      </div>
      <section className="insight-grid">
        <article>
          <Compass size={22} />
          <span>Genre compass</span>
          <strong>{favoriteGenre}</strong>
        </article>
        <article>
          <CalendarClock size={22} />
          <span>Time machine</span>
          <strong>{signatureDecade}</strong>
        </article>
        <article>
          <Star size={22} />
          <span>Rating mood</span>
          <strong>{ratingMood}</strong>
        </article>
        <article>
          <Sparkles size={22} />
          <span>Discovery style</span>
          <strong>{topDiscovery}</strong>
        </article>
      </section>
      <section className="metric-strip">
        <div><strong>{data.movies_watched}</strong><span>Watched</span></div>
        <div><strong>{data.movies_rated}</strong><span>Rated</span></div>
        <div><strong>{data.avg_rating?.toFixed(1) ?? "n/a"}</strong><span>Average rating</span></div>
      </section>
      <section className="analytics-grid">
        <article className="chart-panel">
          <h2>Genre pull</h2>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={genres} dataKey="value" nameKey="name" innerRadius={58} outerRadius={92} paddingAngle={3}>
                {genres.map((_, index) => <Cell key={index} fill={colors[index % colors.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </article>
        <article className="chart-panel">
          <h2>Ratings</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={ratings}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d8ded8" />
              <XAxis dataKey="name" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="value" fill="#0f766e" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>
        <article className="chart-panel">
          <h2>Decades</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={decades}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d8ded8" />
              <XAxis dataKey="name" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="value" fill="#f59e0b" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>
        <article className="chart-panel">
          <h2>Discovery styles</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={clusters}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d8ded8" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="value" fill="#be123c" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </article>
      </section>
    </div>
  );
}
