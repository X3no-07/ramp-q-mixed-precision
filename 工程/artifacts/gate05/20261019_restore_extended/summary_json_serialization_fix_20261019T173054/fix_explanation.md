# Local summary serialization fix

The original external plan has 7772 bytes with 189 CRLF line endings. The unchanged native queue parses it, then save() emits the same complete JSON payload as 7583 bytes with 189 LF line endings. This was independently confirmed by exact CRLF-to-LF byte conversion and by complete object comparison.

The old summary incorrectly required the queue-saved file to have the external source-byte SHA256. The summary now checks the queue-saved file against its own immutable remote receipt and compares its complete canonical JSON payload to the frozen external source. Canonical serialization preserves JSON types and every field/list value. The queue request must still retain the exact external original_plan_sha256; the collected external plan, budget, script, bundle inputs, monitor statuses and build-result file hashes retain their original strict byte checks. Budget, roots, request identities and all native source code remain unchanged.

Before-fix tool/checkpoint and their hashes are retained beside this note. The correction verified all three real completed queue/build identities; precision/smoke receipts that were not yet collected remain PENDING. Gate 0.5 remains INCOMPLETE. No SSH or GPU was used and no raw receipt was modified.
