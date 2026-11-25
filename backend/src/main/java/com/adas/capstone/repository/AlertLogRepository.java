package com.adas.capstone.repository;

import com.adas.capstone.model.AlertLog;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface AlertLogRepository extends MongoRepository<AlertLog, String> {
    List<AlertLog> findBySessionIdOrderByTimestampDesc(String sessionId);
    List<AlertLog> findTop100ByOrderByTimestampDesc();
}
