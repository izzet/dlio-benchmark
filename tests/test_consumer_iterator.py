from dlio_benchmark.data_loader import consumer_iterator


class _Clock:
    def __init__(self):
        self.now = 0


class _Source:
    def __init__(self, clock, waits):
        self.clock = clock
        self.waits = iter(waits)

    def __iter__(self):
        return self

    def __next__(self):
        wait = next(self.waits)
        self.clock.now += wait
        return wait


class _Region:
    def __init__(self, clock, events, name):
        self.clock = clock
        self.events = events
        self.name = name
        self.started = None

    def start(self):
        self.started = self.clock.now

    def stop(self):
        self.events.append((self.name, self.started, self.clock.now))
        self.started = None


class _Fetch:
    def __init__(self, clock, events):
        self.clock = clock
        self.events = events
        self.block = _Region(clock, events, "fetch.block")

    def derive(self, name):
        assert name == "block"
        return self.block

    def iter(self, source, include_block=True):
        assert include_block is False
        source = iter(source)
        while True:
            start = self.clock.now
            try:
                batch = next(source)
            except StopIteration:
                return
            self.events.append(("fetch.iter", start, self.clock.now))
            yield batch


def test_consumer_regions_are_sibling_next_and_body_events(monkeypatch):
    clock = _Clock()
    events = []
    fetch = _Fetch(clock, events)
    fake_ai = type("AI", (), {
        "dataloader": type("DataLoader", (), {"fetch": fetch})(),
    })()
    monkeypatch.setattr(consumer_iterator, "dft_ai", fake_ai)

    batches = consumer_iterator.traced_consumer_batches(_Source(clock, [5, 3]))
    assert next(batches) == 5
    clock.now += 7
    assert next(batches) == 3
    clock.now += 11
    try:
        next(batches)
    except StopIteration:
        pass

    assert events == [
        ("fetch.iter", 0, 5),
        ("fetch.block", 5, 12),
        ("fetch.iter", 12, 15),
        ("fetch.block", 15, 26),
    ]
