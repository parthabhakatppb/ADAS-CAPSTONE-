package com.adas.capstone;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class AdasCapstoneApplication {
    public static void main(String[] args) {
        SpringApplication.run(AdasCapstoneApplication.class, args);
    }
}
