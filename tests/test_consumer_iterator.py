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
        self.regions = {
            "iter": _Region(clock, events, "fetch.iter"),
            "block": _Region(clock, events, "fetch.block"),
        }

    def derive(self, name):
        return self.regions[name]


def test_consumer_regions_are_sibling_next_and_body_events(monkeypatch):
    clock = _Clock()
    events = []
    fetch = _Fetch(clock, events)
    fake_ai = type("AI", (), {
        "dataloader": type("DataLoader", (), {"fetch": fetch})(),
    })()
    monkeypatch.setattr(consumer_iterator, "dft_ai", fake_ai)

    boundary = consumer_iterator.ConsumerBoundary(
        _Source(clock, [5, 3]), clock=lambda: clock.now
    )
    assert boundary.next_batch() == 5
    with boundary.consumer_body():
        clock.now += 7
    assert boundary.next_batch() == 3
    with boundary.consumer_body():
        clock.now += 11
    try:
        boundary.next_batch()
    except StopIteration:
        pass

    assert events == [
        ("fetch.iter", 0, 5),
        ("fetch.block", 5, 12),
        ("fetch.iter", 12, 15),
        ("fetch.block", 15, 26),
        # The exhausted source call is traced but is not charged as a batch stall.
        ("fetch.iter", 26, 26),
    ]
    assert boundary.totals == (8, 2)


class _Tracer:
    def __init__(self):
        self.events = []

    def get_time(self):
        return 1_000_000

    def log_event(self, **event):
        self.events.append(event)


class _TracerFactory:
    tracer = _Tracer()

    @classmethod
    def get_instance(cls):
        return cls.tracer


def test_window_stall_is_one_accumulated_main_thread_event(monkeypatch):
    clock = _Clock()
    fetch = _Fetch(clock, [])
    fake_ai = type("AI", (), {
        "dataloader": type("DataLoader", (), {"fetch": fetch})(),
    })()
    monkeypatch.setattr(consumer_iterator, "dft_ai", fake_ai)
    boundary = consumer_iterator.ConsumerBoundary(
        _Source(clock, [0.004, 0.006]), clock=lambda: clock.now
    )
    boundary.next_batch()
    boundary.next_batch()

    _TracerFactory.tracer = _Tracer()
    emitted = boundary.emit_window_stall(2, 8, tracer_factory=_TracerFactory)

    assert emitted == 0.01
    assert boundary.pending_stall_seconds == 0.0
    assert len(_TracerFactory.tracer.events) == 1
    event = _TracerFactory.tracer.events[0]
    assert (event["cat"], event["name"], event["duration"]) == (
        "stall", "stall.storage", 10_000
    )
    assert event["int_args"] == {"epoch": (0, 2), "step": (0, 8)}


def test_zero_window_stall_is_emitted_as_present_zero(monkeypatch):
    clock = _Clock()
    fetch = _Fetch(clock, [])
    fake_ai = type("AI", (), {
        "dataloader": type("DataLoader", (), {"fetch": fetch})(),
    })()
    monkeypatch.setattr(consumer_iterator, "dft_ai", fake_ai)
    boundary = consumer_iterator.ConsumerBoundary([], clock=lambda: clock.now)

    _TracerFactory.tracer = _Tracer()
    assert boundary.emit_window_stall(3, 9, tracer_factory=_TracerFactory) == 0.0

    assert len(_TracerFactory.tracer.events) == 1
    event = _TracerFactory.tracer.events[0]
    assert (event["cat"], event["name"], event["duration"]) == (
        "stall", "stall.storage", 0
    )
