pipeline {
    agent any
    stages {
        stage('build') {
            steps {
                sh 'build.sh'
            }
        }
        stage('test') {
            steps {
                sh 'test.sh'
            }
        }
    }
}
