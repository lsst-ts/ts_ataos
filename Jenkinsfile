pipeline {
    agent any
    environment {
        container_name = "c_${BUILD_ID}_${JENKINS_NODE_COOKIE}"
        image_tag = "master"
        user_ci = credentials('lsst-io')
        work_branches = "${GIT_BRANCH} ${CHANGE_BRANCH} develop"
    }

    stages {
        stage("Pull docker image") {
            steps {
                script {
                    sh """
                    docker pull lsstts/salobj:\${image_tag}
                    """
                }
            }
        }
        stage("Prepare Workspace") {
            steps {
                script {
                    sh """
                    chmod -R a+rw \${WORKSPACE} || echo "Failed to set workspace mode"
                    """
                }
            }
        }
        stage("Run container") {
            steps {
                script {
                    sh """
                    container=\$(docker run -v \${WORKSPACE}:/home/saluser/repo/ -td --rm --name \${container_name} -e LTD_USERNAME=\${user_ci_USR} -e LTD_PASSWORD=\${user_ci_PSW} lsstts/salobj:\${image_tag})
                    """
                }
            }
        }
        stage("Clone/Setup configuration repo") {
            steps {
                script {
                    sh """
                    docker exec -u saluser \${container_name} sh -c \"source ~/.setup.sh && cd /home/saluser/repos/ts_config_attcs && /home/saluser/.checkout_repo.sh \${work_branches}\"
                    """
                }
            }
        }
        stage("Checkout xml") {
            steps {
                script {
                    sh """
                    docker exec -u saluser \${container_name} sh -c \"source ~/.setup.sh && cd /home/saluser/repos/ts_xml && /home/saluser/.checkout_repo.sh \${work_branches}\"
                    """
                }
            }
        }
        stage("Build IDL Files") {
            steps {
                script {
                    sh """
                    docker exec -u saluser \${container_name} sh -c \"source ~/.setup.sh && source ~/.bashrc && make_idl_files.py ATAOS ATMCS ATPneumatics ATHexapod ATSpectrograph ATCamera ATPtg\"
                    """
                }
            }
        }
        stage("Running tests") {
            steps {
                script {
                    sh """
                    docker exec -u saluser \${container_name} sh -c \"source ~/.setup.sh && cd repo && eups declare -r . -t saluser && setup ts_ataos -t saluser && scons\"
                    """
                }
            }
        }
    }
    post {
        always {
            // The path of xml needed by JUnit is relative to
            // the workspace.
            junit 'tests/.tests/*.xml'

            // Publish the HTML report
            publishHTML (target: [
                allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: true,
                reportDir: 'tests/.tests/pytest-ts_ataos.xml-htmlcov/',
                reportFiles: 'index.html',
                reportName: "Coverage Report"
              ])
        }
        cleanup {
            sh """
                docker exec -u root --privileged \${container_name} sh -c \"chmod -R a+rw /home/saluser/repo/ \"
                docker stop \${container_name}
            """
            deleteDir()
        }
    }
}
