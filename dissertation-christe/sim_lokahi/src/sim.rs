use crate::config::Config;
use crate::constants;
use crate::storage::{Storage, StorageItem};
use crate::{config, storage};
use rand;
use rand::prelude::ThreadRng;
use rand::Rng;
use std::io::{BufWriter, Write};

#[inline]
fn percent_chance(chance: f64, rng: &mut ThreadRng) -> bool {
    rng.gen_range(0.0, 1.0) < chance
}

#[inline]
pub fn fmt_size_mb(total_bytes: usize) -> String {
    format!("{:.*}MB", 2, total_bytes as f64 / 1_000_000.0)
}

#[inline]
pub fn fmt_percent(percent: f64) -> String {
    format!("{:.*}%", 2, percent * 100.0)
}

pub struct Simulation {
    rng: ThreadRng,
    storage: Storage,
    buf_writer: BufWriter<std::fs::File>,
    conf: Config,

    total_storage_items: usize,
    total_samples: usize,
    total_orphaned_samples: usize,
    total_event_samples: usize,
    total_incident_samples: usize,
    total_phenomena_samples: usize,
    total_measurements: usize,
    total_orphaned_measurements: usize,
    total_event_measurements: usize,
    total_phenomena_measurements: usize,
    total_incident_measurements: usize,
    total_trends: usize,
    total_orphaned_trends: usize,
    total_event_trends: usize,
    total_incident_trends: usize,
    total_phenomena_trends: usize,
    total_events: usize,
    total_orphaned_events: usize,
    total_incident_events: usize,
    total_phenomena_events: usize,
    total_incidents: usize,
    total_orphaned_incidents: usize,
    total_phenomena_incidents: usize,
    total_phenomena: usize,
}

impl Simulation {
    pub fn new(conf: Config) -> Simulation {
        let file = std::fs::File::create(conf.out_file.clone()).unwrap();
        let buf_writer = BufWriter::new(file);
        Simulation {
            rng: rand::thread_rng(),
            storage: Storage::new(),
            buf_writer,
            conf,
            total_storage_items: 0,
            total_samples: 0,
            total_orphaned_samples: 0,
            total_event_samples: 0,
            total_incident_samples: 0,
            total_phenomena_samples: 0,
            total_measurements: 0,
            total_orphaned_measurements: 0,
            total_event_measurements: 0,
            total_phenomena_measurements: 0,
            total_incident_measurements: 0,
            total_trends: 0,
            total_orphaned_trends: 0,
            total_event_trends: 0,
            total_incident_trends: 0,
            total_phenomena_trends: 0,
            total_events: 0,
            total_orphaned_events: 0,
            total_incident_events: 0,
            total_phenomena_events: 0,
            total_incidents: 0,
            total_orphaned_incidents: 0,
            total_phenomena_incidents: 0,
            total_phenomena: 0,
        }
    }

    fn make_sample(
        &mut self,
        time: usize,
        is_event: bool,
        is_incident: bool,
        is_phenomena: bool,
    ) -> StorageItem {
        self.total_storage_items += 1;
        self.total_samples += 1;
        let ttl = if !is_event && !is_incident {
            self.total_orphaned_samples += 1;
            time + self.conf.samples_ttl
        } else if is_event {
            self.total_event_samples += 1;
            time + self.conf.events_ttl
        } else {
            self.total_incident_samples += 1;
            time + self.conf.incidents_ttl
        };

        let is_event = if is_event { Some(is_event) } else { None };

        let is_incident = if is_incident { Some(is_incident) } else { None };

        let is_phenomena = if is_phenomena {
            Some(is_phenomena)
        } else {
            None
        };

        storage::StorageItem::new_sample(time, ttl, is_event, is_incident, is_phenomena)
    }

    fn make_measurement(
        &mut self,
        time: usize,
        is_event: bool,
        is_incident: bool,
        is_phenomena: bool,
    ) -> StorageItem {
        self.total_storage_items += 1;
        self.total_measurements += 1;

        let ttl = if is_phenomena {
            self.total_phenomena_measurements += 1;
            time + self.conf.phenomena_ttl
        } else if is_incident {
            self.total_incident_measurements += 1;
            time + self.conf.incidents_ttl
        } else if is_event {
            self.total_event_measurements += 1;
            time + self.conf.events_ttl
        } else {
            self.total_orphaned_measurements += 1;
            time + self.conf.measurements_ttl
        };

        let is_event = if is_event { Some(is_event) } else { None };

        let is_incident = if is_incident { Some(is_incident) } else { None };

        let is_phenomena = if is_phenomena {
            Some(is_phenomena)
        } else {
            None
        };

        storage::StorageItem::new_measurement(time, ttl, is_event, is_incident, is_phenomena)
    }

    fn make_trend(
        &mut self,
        time: usize,
        is_event: bool,
        is_incident: bool,
        is_phenomena: bool,
    ) -> StorageItem {
        self.total_storage_items += 1;
        self.total_trends += 1;

        let ttl = if is_phenomena {
            self.total_phenomena_trends += 1;
            time + self.conf.phenomena_ttl
        } else if is_incident {
            self.total_incident_trends += 1;
            time + self.conf.incidents_ttl
        } else if is_event {
            self.total_event_trends += 1;
            time + self.conf.events_ttl
        } else {
            self.total_orphaned_trends += 1;
            time + self.conf.trends_ttl
        };

        let is_event = if is_event { Some(is_event) } else { None };

        let is_incident = if is_incident { Some(is_incident) } else { None };

        let is_phenomena = if is_phenomena {
            Some(is_phenomena)
        } else {
            None
        };

        storage::StorageItem::new_trend(time, ttl, is_event, is_incident, is_phenomena)
    }

    fn make_event(&mut self, time: usize, is_incident: bool, is_phenomena: bool) -> StorageItem {
        self.total_storage_items += 1;
        self.total_events += 1;

        let ttl = if is_phenomena {
            self.total_phenomena_events += 1;
            time + self.conf.phenomena_ttl
        } else if is_incident {
            self.total_incident_events += 1;
            time + self.conf.incidents_ttl
        } else {
            self.total_orphaned_events += 1;
            time + self.conf.events_ttl
        };

        let is_incident = if is_incident { Some(is_incident) } else { None };

        let is_phenomena = if is_phenomena {
            Some(is_phenomena)
        } else {
            None
        };

        storage::StorageItem::new_event(time, ttl, Some(false), is_incident, is_phenomena)
    }

    fn make_incident(&mut self, time: usize, is_phenomena: bool) -> StorageItem {
        self.total_storage_items += 1;
        self.total_incidents += 1;

        let ttl = if is_phenomena {
            self.total_phenomena_incidents += 1;
            time + self.conf.phenomena_ttl
        } else {
            self.total_orphaned_incidents += 1;
            time + self.conf.incidents_ttl
        };

        let is_phenomena = if is_phenomena {
            Some(is_phenomena)
        } else {
            None
        };

        storage::StorageItem::new_incident(time, ttl, Some(false), Some(false), is_phenomena)
    }

    fn make_phenomena(&mut self, time: usize) -> StorageItem {
        self.total_storage_items += 1;
        self.total_phenomena += 1;

        storage::StorageItem::new_phenomena(
            time,
            time + self.conf.phenomena_ttl,
            Some(false),
            Some(false),
            Some(false),
        )
    }

    fn write_to_file(&mut self, time: usize) {
        let storage_stats = self.storage.stat_storage_items(None, None, None, None);
        let sample_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::MetaSample(
                constants::ESTIMATED_BYTES_PER_META_SAMPLE_8000,
            )),
            None,
            None,
            None,
        );
        let measurement_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(
                constants::ESTIMATED_BYTES_PER_MEASUREMENT,
            )),
            None,
            None,
            None,
        );
        let measurement_orphaned_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(
                constants::ESTIMATED_BYTES_PER_MEASUREMENT,
            )),
            Some(false),
            Some(false),
            Some(false),
        );
        let measurement_event_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(
                constants::ESTIMATED_BYTES_PER_MEASUREMENT,
            )),
            Some(true),
            Some(false),
            Some(false),
        );
        let measurement_incident_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(
                constants::ESTIMATED_BYTES_PER_MEASUREMENT,
            )),
            Some(false),
            Some(true),
            Some(false),
        );
        let measurement_phenomena_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(
                constants::ESTIMATED_BYTES_PER_MEASUREMENT,
            )),
            Some(false),
            Some(false),
            Some(true),
        );
        let trends_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(
                constants::ESTIMATED_BYTES_PER_TREND,
            )),
            None,
            None,
            None,
        );
        let trends_orphaned_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(
                constants::ESTIMATED_BYTES_PER_TREND,
            )),
            Some(false),
            Some(false),
            Some(false),
        );
        let trends_event_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(
                constants::ESTIMATED_BYTES_PER_TREND,
            )),
            Some(true),
            Some(false),
            Some(false),
        );
        let trends_incident_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(
                constants::ESTIMATED_BYTES_PER_TREND,
            )),
            Some(false),
            Some(true),
            Some(false),
        );
        let trends_phenomena_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(
                constants::ESTIMATED_BYTES_PER_TREND,
            )),
            Some(false),
            Some(false),
            Some(true),
        );
        let event_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Event(
                constants::ESTIMATED_BYTES_PER_EVENT_8000,
            )),
            None,
            None,
            None,
        );
        let event_orphaned_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Event(
                constants::ESTIMATED_BYTES_PER_EVENT_8000,
            )),
            Some(false),
            Some(false),
            Some(false),
        );
        let event_incident_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Event(
                constants::ESTIMATED_BYTES_PER_EVENT_8000,
            )),
            Some(false),
            Some(true),
            Some(false),
        );
        let event_phenomena_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Event(
                constants::ESTIMATED_BYTES_PER_EVENT_8000,
            )),
            Some(false),
            Some(false),
            Some(true),
        );
        let incident_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Incident(
                constants::ESTIMATED_BYTES_PER_INCIDENT_8000,
            )),
            Some(false),
            Some(false),
            Some(false),
        );
        let incident_phenomena_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Incident(
                constants::ESTIMATED_BYTES_PER_INCIDENT_8000,
            )),
            Some(false),
            Some(false),
            Some(true),
        );
        let phenomena_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Phenomena(
                constants::ESTIMATED_BYTES_PER_PHENOMENA,
            )),
            None,
            None,
            None,
        );

        let items: Vec<usize> = vec![
            time,
            sample_stats.items * constants::SAMPLES_PER_SECOND_8000,
            sample_stats.total_bytes,
            measurement_stats.items,
            measurement_stats.total_bytes,
            measurement_orphaned_stats.items,
            measurement_orphaned_stats.total_bytes,
            measurement_event_stats.items,
            measurement_event_stats.total_bytes,
            measurement_incident_stats.items,
            measurement_incident_stats.total_bytes,
            measurement_phenomena_stats.items,
            measurement_phenomena_stats.total_bytes,
            trends_stats.items,
            trends_stats.total_bytes,
            trends_orphaned_stats.items,
            trends_orphaned_stats.total_bytes,
            trends_event_stats.items,
            trends_event_stats.total_bytes,
            trends_incident_stats.items,
            trends_incident_stats.total_bytes,
            trends_phenomena_stats.items,
            trends_phenomena_stats.total_bytes,
            event_stats.items,
            event_stats.total_bytes,
            event_orphaned_stats.items,
            event_orphaned_stats.total_bytes,
            event_incident_stats.items,
            event_incident_stats.total_bytes,
            event_phenomena_stats.items,
            event_phenomena_stats.total_bytes,
            incident_stats.items,
            incident_stats.total_bytes,
            incident_phenomena_stats.items,
            incident_phenomena_stats.total_bytes,
            phenomena_stats.items,
            phenomena_stats.total_bytes,
            storage_stats.total_bytes,
            sample_stats.total_bytes,
            measurement_stats.total_bytes + trends_stats.total_bytes,
            event_stats.total_bytes,
            incident_stats.total_bytes,
            phenomena_stats.total_bytes,
        ];
        let items: Vec<String> = items.iter().map(|i| i.to_string()).collect();
        let line: String = format!("{}\n", items.join(","));
        let res = self.buf_writer.write(line.as_bytes()).unwrap();
    }

    fn display_info(&mut self, time: usize) {
        let storage_stats = self.storage.stat_storage_items(None, None, None, None);
        let sample_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::MetaSample(
                constants::ESTIMATED_BYTES_PER_META_SAMPLE_8000,
            )),
            None,
            None,
            None,
        );
        let measurement_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(
                constants::ESTIMATED_BYTES_PER_MEASUREMENT,
            )),
            None,
            None,
            None,
        );
        let measurement_orphaned_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(
                constants::ESTIMATED_BYTES_PER_MEASUREMENT,
            )),
            Some(false),
            Some(false),
            Some(false),
        );
        let measurement_event_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(
                constants::ESTIMATED_BYTES_PER_MEASUREMENT,
            )),
            Some(true),
            Some(false),
            Some(false),
        );
        let measurement_incident_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(
                constants::ESTIMATED_BYTES_PER_MEASUREMENT,
            )),
            Some(false),
            Some(true),
            Some(false),
        );
        let measurement_phenomena_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(
                constants::ESTIMATED_BYTES_PER_MEASUREMENT,
            )),
            Some(false),
            Some(false),
            Some(true),
        );
        let trends_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(
                constants::ESTIMATED_BYTES_PER_TREND,
            )),
            None,
            None,
            None,
        );
        let trends_orphaned_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(
                constants::ESTIMATED_BYTES_PER_TREND,
            )),
            Some(false),
            Some(false),
            Some(false),
        );
        let trends_event_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(
                constants::ESTIMATED_BYTES_PER_TREND,
            )),
            Some(true),
            Some(false),
            Some(false),
        );
        let trends_incident_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(
                constants::ESTIMATED_BYTES_PER_TREND,
            )),
            Some(false),
            Some(true),
            Some(false),
        );
        let trends_phenomena_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(
                constants::ESTIMATED_BYTES_PER_TREND,
            )),
            Some(false),
            Some(false),
            Some(true),
        );
        let event_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Event(
                constants::ESTIMATED_BYTES_PER_EVENT_8000,
            )),
            None,
            None,
            None,
        );
        let event_orphaned_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Event(
                constants::ESTIMATED_BYTES_PER_EVENT_8000,
            )),
            Some(false),
            Some(false),
            Some(false),
        );
        let event_incident_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Event(
                constants::ESTIMATED_BYTES_PER_EVENT_8000,
            )),
            Some(false),
            Some(true),
            Some(false),
        );
        let event_phenomena_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Event(
                constants::ESTIMATED_BYTES_PER_EVENT_8000,
            )),
            Some(false),
            Some(false),
            Some(true),
        );
        let incident_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Incident(
                constants::ESTIMATED_BYTES_PER_INCIDENT_8000,
            )),
            Some(false),
            Some(false),
            Some(false),
        );
        let incident_phenomena_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Incident(
                constants::ESTIMATED_BYTES_PER_INCIDENT_8000,
            )),
            Some(false),
            Some(false),
            Some(true),
        );
        let phenomena_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Phenomena(
                constants::ESTIMATED_BYTES_PER_PHENOMENA,
            )),
            None,
            None,
            None,
        );

        println!(
            "{}/{} {:.*}%",
            time,
            self.conf.ticks,
            1,
            time as f64 / self.conf.ticks as f64 * 100.0
        );

        println!(
            "\ttotal_samples={} {}",
            sample_stats.items * 12_000,
            sample_stats.fmt_size_mb()
        );

        println!(
            "\ttotal_measurements={} {} orphaned_measurements={} {} {} event_measurements={} {} {} incident_measurements={} {} {} phenomena_measurements={} {} {}",
            measurement_stats.items,
            measurement_stats.fmt_size_mb(),
            measurement_orphaned_stats.items,
            measurement_orphaned_stats.fmt_percent(),
            measurement_orphaned_stats.fmt_size_mb(),
            measurement_event_stats.items,
            measurement_event_stats.fmt_percent(),
            measurement_event_stats.fmt_size_mb(),
            measurement_incident_stats.items,
            measurement_incident_stats.fmt_percent(),
            measurement_incident_stats.fmt_size_mb(),
            measurement_phenomena_stats.items,
            measurement_phenomena_stats.fmt_percent(),
            measurement_phenomena_stats.fmt_size_mb()
        );

        println!(
            "\ttotal_trends={} {} orphaned_trends={} {} {} event_trends={} {} {} incident_trends={} {} {} phenomena_trends={} {} {}",
            trends_stats.items,
            trends_stats.fmt_size_mb(),
            trends_orphaned_stats.items,
            trends_orphaned_stats.fmt_percent(),
            trends_orphaned_stats.fmt_size_mb(),
            trends_event_stats.items,
            trends_event_stats.fmt_percent(),
            trends_event_stats.fmt_size_mb(),
            trends_incident_stats.items,
            trends_incident_stats.fmt_percent(),
            trends_incident_stats.fmt_size_mb(),
            trends_phenomena_stats.items,
            trends_phenomena_stats.fmt_percent(),
            trends_phenomena_stats.fmt_size_mb()
        );

        println!(
            "\ttotal_events={} {} orphaned_events={} {} {} incident_events={} {} {} phenomena_events={} {} {}",
            event_stats.items,
            event_stats.fmt_size_mb(),
            event_orphaned_stats.items,
            event_orphaned_stats.fmt_percent(),
            event_orphaned_stats.fmt_size_mb(),
            event_incident_stats.items,
            event_incident_stats.fmt_percent(),
            event_incident_stats.fmt_size_mb(),
            event_phenomena_stats.items,
            event_phenomena_stats.fmt_percent(),
            event_phenomena_stats.fmt_size_mb()
        );

        println!(
            "\ttotal_incidents={} {} phenomena_incidents={} {} {}",
            incident_stats.items,
            incident_stats.fmt_size_mb(),
            incident_phenomena_stats.items,
            incident_phenomena_stats.fmt_percent(),
            incident_phenomena_stats.fmt_size_mb()
        );

        println!(
            "\ttotal_phenomena={} {}",
            phenomena_stats.items,
            phenomena_stats.fmt_size_mb(),
        );

        println!(
            "\ttotal_laha={} total_iml={} total_aml={} total_dl={} total_il={} total_pl={}",
            storage_stats.fmt_size_mb(),
            sample_stats.fmt_size_mb(),
            fmt_size_mb(measurement_stats.total_bytes + trends_stats.total_bytes),
            event_stats.fmt_size_mb(),
            incident_stats.fmt_size_mb(),
            phenomena_stats.fmt_size_mb()
        );
    }

    fn display_summary(&self) {
        let total_samples = self.total_samples * constants::SAMPLES_PER_SECOND_8000;
        println!(
            "total_samples={} {}",
            total_samples,
            fmt_size_mb(total_samples * constants::BYTES_PER_SAMPLE)
        );

        println!(
            "total_measurements={} {} orphaned_measurements={} {} {} event_measurements={} {} {} incident_measurements={} {} {} phenomena_measurements={} {} {}",
            self.total_measurements,
            fmt_size_mb(self.total_measurements * constants::ESTIMATED_BYTES_PER_MEASUREMENT),
            self.total_orphaned_measurements,
            fmt_percent(self.total_orphaned_measurements as f64 / self.total_measurements as f64),
            fmt_size_mb(self.total_orphaned_measurements * constants::ESTIMATED_BYTES_PER_MEASUREMENT),
            self.total_event_measurements,
            fmt_percent(self.total_event_measurements as f64 / self.total_measurements as f64),
            fmt_size_mb(self.total_event_measurements * constants::ESTIMATED_BYTES_PER_MEASUREMENT),
            self.total_incident_measurements,
            fmt_percent(self.total_incident_measurements as f64 / self.total_measurements as f64),
            fmt_size_mb(self.total_incident_measurements * constants::ESTIMATED_BYTES_PER_MEASUREMENT),
            self.total_phenomena_measurements,
            fmt_percent(self.total_phenomena_measurements as f64 / self.total_measurements as f64),
            fmt_size_mb(self.total_phenomena_measurements * constants::ESTIMATED_BYTES_PER_MEASUREMENT)
        );

        println!(
            "total_trends={} {} orphaned_trends={} {} {} event_trends={} {} {} incident_trends={} {} {} incident_phenomena={} {} {}",
            self.total_trends,
            fmt_size_mb(self.total_trends * constants::ESTIMATED_BYTES_PER_TREND),
            self.total_orphaned_trends,
            fmt_percent(self.total_orphaned_trends as f64 / self.total_trends as f64),
            fmt_size_mb(self.total_orphaned_trends * constants::ESTIMATED_BYTES_PER_TREND),
            self.total_event_trends,
            fmt_percent(self.total_event_trends as f64 / self.total_trends as f64),
            fmt_size_mb(self.total_event_trends * constants::ESTIMATED_BYTES_PER_TREND),
            self.total_incident_trends,
            fmt_percent(self.total_incident_trends as f64 / self.total_trends as f64),
            fmt_size_mb(self.total_incident_trends * constants::ESTIMATED_BYTES_PER_TREND),
            self.total_phenomena_trends,
            fmt_percent(self.total_phenomena_trends as f64 / self.total_trends as f64),
            fmt_size_mb(self.total_phenomena_trends * constants::ESTIMATED_BYTES_PER_TREND)
        );

        println!(
            "total_events={} {} orphaned_events={} {} {} incident_events={} {} {} phenomena_events={} {} {}",
            self.total_events,
            fmt_size_mb(self.total_events * constants::ESTIMATED_BYTES_PER_EVENT_8000),
            self.total_orphaned_events,
            fmt_percent(self.total_orphaned_events as f64 / self.total_events as f64),
            fmt_size_mb(self.total_orphaned_events * constants::ESTIMATED_BYTES_PER_EVENT_8000),
            self.total_incident_events,
            fmt_percent(self.total_incident_events as f64 / self.total_events as f64),
            fmt_size_mb(self.total_incident_events * constants::ESTIMATED_BYTES_PER_EVENT_8000),
            self.total_phenomena_events,
            fmt_percent(self.total_phenomena_events as f64 / self.total_events as f64),
            fmt_size_mb(self.total_phenomena_events * constants::ESTIMATED_BYTES_PER_EVENT_8000)
        );

        println!(
            "total_incidents={} {} phenomena_incidents={} {} {}",
            self.total_incidents,
            fmt_size_mb(self.total_incidents * constants::ESTIMATED_BYTES_PER_INCIDENT_8000),
            self.total_phenomena_incidents,
            fmt_percent(self.total_phenomena_incidents as f64 / self.total_incidents as f64),
            fmt_size_mb(
                self.total_phenomena_incidents * constants::ESTIMATED_BYTES_PER_INCIDENT_8000
            )
        );

        println!(
            "total_incidents={} {}",
            self.total_phenomena,
            fmt_size_mb(self.total_phenomena * constants::ESTIMATED_BYTES_PER_PHENOMENA),
        );

        println!("total_storage_items={}", self.total_storage_items);
    }

    pub fn run_simulation(&mut self) {
        let mut storage_items_per_tick: Vec<storage::StorageItem> = vec![];

        for i in 0..self.conf.ticks {
            storage_items_per_tick.clear();

            // Chance of producing samples and trends
            if i % constants::META_SAMPLE_8000_LEN == 0 {
                storage_items_per_tick.push(self.make_sample(i, false, false, false));
                if percent_chance(
                    constants::ESTIMATED_PERCENT_EVENT_DATA_DURATION,
                    &mut self.rng,
                ) {
                    storage_items_per_tick.push(self.make_trend(i, true, false, false));
                } else if percent_chance(
                    constants::ESTIMATED_PERCENT_INCIDENT_DATA_DURATION,
                    &mut self.rng,
                ) {
                    storage_items_per_tick.push(self.make_trend(i, false, true, false));
                } else if percent_chance(
                    constants::ESTIMATED_PERCENT_INCIDENT_DATA_DURATION,
                    &mut self.rng,
                ) {
                    storage_items_per_tick.push(self.make_trend(i, false, false, true));
                } else {
                    storage_items_per_tick.push(self.make_trend(i, false, false, false));
                }
            }

            // Chance of producing an event
            if percent_chance(constants::ESTIMATED_EVENTS_PER_SECOND, &mut self.rng) {
                storage_items_per_tick.push(self.make_event(i, false, false));
            }

            // Chance of producing an incident
            if percent_chance(constants::ESTIMATED_INCIDENTS_PER_SECOND, &mut self.rng) {
                storage_items_per_tick.push(self.make_incident(i, false));
            }

            // Chance og producing a phenomena
            if percent_chance(constants::ESTIMATED_INCIDENTS_PER_SECOND, &mut self.rng) {
                storage_items_per_tick.push(self.make_phenomena(i));
            }

            self.storage.add_many(&mut storage_items_per_tick, i);

            if i % self.conf.print_info_every_n_ticks == 0 {
                self.display_info(i);
            }

            if i % self.conf.write_info_every_n_ticks == 0 {
                self.write_to_file(i);
            }
        }

        self.display_summary();
        self.buf_writer.flush().unwrap();
    }
}
