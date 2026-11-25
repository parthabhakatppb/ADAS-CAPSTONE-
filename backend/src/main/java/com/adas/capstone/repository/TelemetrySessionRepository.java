package com.adas.capstone.repository;

import com.adas.capstone.model.TelemetrySession;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface TelemetrySessionRepository extends MongoRepository<TelemetrySession, String> {
    List<TelemetrySession> findTop20ByOrderByStartTimeDesc();
}
