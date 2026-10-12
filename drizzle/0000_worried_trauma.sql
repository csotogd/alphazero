CREATE TABLE `rooms` (
	`code` text PRIMARY KEY NOT NULL,
	`state` text NOT NULL,
	`host_hash` text NOT NULL,
	`guest_hash` text,
	`host_name` text NOT NULL,
	`guest_name` text,
	`revision` integer DEFAULT 0 NOT NULL,
	`rematch` integer DEFAULT 0 NOT NULL,
	`created_at` integer NOT NULL
);
