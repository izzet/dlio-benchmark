"""Trace the two halves of a data-loader consumer iteration separately."""

from dlio_benchmark.utils.utility import dft_ai


def traced_consumer_batches(source):
    """Yield *source* with exact, sibling consumer-boundary regions.

    ``fetch.iter`` covers only the call which obtains the next batch from the
    source iterator. ``fetch.block`` covers only the caller's body between
    receiving that batch and requesting the next one.  Keeping the two as
    siblings is the one-level selection contract used by the DLIO analyzer
    preset.
    """
    fetch = dft_ai.dataloader.fetch
    consumer_body = fetch.derive("block")

    # The tracer's iterator helper starts fetch.iter immediately before it
    # advances source and stops it immediately after a batch is returned.
    # Disable its built-in block so the loader's own bookkeeping between that
    # return and the outward yield is not charged to the consumer body.
    for batch in fetch.iter(source, include_block=False):
        consumer_body.start()
        try:
            yield batch
        finally:
            consumer_body.stop()
