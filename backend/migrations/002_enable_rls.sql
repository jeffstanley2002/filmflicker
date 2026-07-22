ALTER TABLE cinematch_v2.watched ENABLE ROW LEVEL SECURITY;
ALTER TABLE cinematch_v2.ratings ENABLE ROW LEVEL SECURITY;
ALTER TABLE cinematch_v2.not_interested ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS watched_owner_access ON cinematch_v2.watched;
CREATE POLICY watched_owner_access
    ON cinematch_v2.watched
    FOR ALL
    TO authenticated
    USING (user_id = (SELECT auth.uid()))
    WITH CHECK (user_id = (SELECT auth.uid()));

DROP POLICY IF EXISTS ratings_owner_access ON cinematch_v2.ratings;
CREATE POLICY ratings_owner_access
    ON cinematch_v2.ratings
    FOR ALL
    TO authenticated
    USING (user_id = (SELECT auth.uid()))
    WITH CHECK (user_id = (SELECT auth.uid()));

DROP POLICY IF EXISTS not_interested_owner_access ON cinematch_v2.not_interested;
CREATE POLICY not_interested_owner_access
    ON cinematch_v2.not_interested
    FOR ALL
    TO authenticated
    USING (user_id = (SELECT auth.uid()))
    WITH CHECK (user_id = (SELECT auth.uid()));
