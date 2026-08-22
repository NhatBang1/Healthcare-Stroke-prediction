BEGIN;

SELECT pg_advisory_xact_lock(hashtext('strokeguard:migration:002'));

ALTER TABLE strokeguard.patient_profiles
    ADD COLUMN IF NOT EXISTS pregnant BOOLEAN;

INSERT INTO strokeguard.schema_migrations (version, description)
VALUES ('002', 'Add pregnancy context for safe blood-pressure classification')
ON CONFLICT (version) DO NOTHING;

COMMIT;
