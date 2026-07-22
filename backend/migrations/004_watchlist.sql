CREATE TABLE IF NOT EXISTS cinematch_v2.watchlist (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    movie_id INTEGER NOT NULL,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(user_id, movie_id)
);

ALTER TABLE cinematch_v2.watchlist ENABLE ROW LEVEL SECURITY;

CREATE POLICY watchlist_owner_access
    ON cinematch_v2.watchlist
    FOR ALL
    TO authenticated
    USING (user_id = (SELECT auth.uid()))
    WITH CHECK (user_id = (SELECT auth.uid()));

CREATE INDEX IF NOT EXISTS watchlist_user_time_idx
    ON cinematch_v2.watchlist (user_id, added_at DESC);
