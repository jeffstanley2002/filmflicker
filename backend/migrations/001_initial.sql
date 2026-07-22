CREATE SCHEMA IF NOT EXISTS cinematch_v2;

CREATE TABLE IF NOT EXISTS cinematch_v2.watched (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    movie_id INTEGER NOT NULL,
    watched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS cinematch_v2.ratings (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    movie_id INTEGER NOT NULL,
    rating REAL NOT NULL CHECK (rating BETWEEN 0.5 AND 5.0),
    rated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS cinematch_v2.not_interested (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    movie_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(user_id, movie_id)
);

ALTER TABLE cinematch_v2.watched ENABLE ROW LEVEL SECURITY;
ALTER TABLE cinematch_v2.ratings ENABLE ROW LEVEL SECURITY;
ALTER TABLE cinematch_v2.not_interested ENABLE ROW LEVEL SECURITY;

CREATE POLICY watched_owner_access
    ON cinematch_v2.watched
    FOR ALL
    TO authenticated
    USING (user_id = (SELECT auth.uid()))
    WITH CHECK (user_id = (SELECT auth.uid()));

CREATE POLICY ratings_owner_access
    ON cinematch_v2.ratings
    FOR ALL
    TO authenticated
    USING (user_id = (SELECT auth.uid()))
    WITH CHECK (user_id = (SELECT auth.uid()));

CREATE POLICY not_interested_owner_access
    ON cinematch_v2.not_interested
    FOR ALL
    TO authenticated
    USING (user_id = (SELECT auth.uid()))
    WITH CHECK (user_id = (SELECT auth.uid()));

CREATE INDEX IF NOT EXISTS watched_user_time_idx
    ON cinematch_v2.watched (user_id, watched_at DESC);
CREATE INDEX IF NOT EXISTS ratings_user_idx
    ON cinematch_v2.ratings (user_id);
CREATE INDEX IF NOT EXISTS not_interested_user_idx
    ON cinematch_v2.not_interested (user_id);
