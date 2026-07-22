DO $$
DECLARE
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY['profiles', 'watched', 'ratings', 'not_interested']
    LOOP
        IF to_regclass('public.' || quote_ident(table_name)) IS NOT NULL THEN
            EXECUTE 'ALTER TABLE public.' || quote_ident(table_name) || ' ENABLE ROW LEVEL SECURITY';
            EXECUTE 'REVOKE ALL ON TABLE public.' || quote_ident(table_name) || ' FROM anon, authenticated';
        END IF;
    END LOOP;
END $$;
