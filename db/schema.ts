import { sqliteTable, text, integer } from 'drizzle-orm/sqlite-core';
export const rooms = sqliteTable('rooms', {
  code: text('code').primaryKey(),
  state: text('state').notNull(),
  hostHash: text('host_hash').notNull(),
  guestHash: text('guest_hash'),
  hostName: text('host_name').notNull(),
  guestName: text('guest_name'),
  revision: integer('revision').notNull().default(0),
  rematch: integer('rematch').notNull().default(0),
  createdAt: integer('created_at').notNull(),
});
