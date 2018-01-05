#!/p/php/php
Before

<?php
Print "Hello";
$ip = getenv('REMOTE_HOST');
Print $ip;
Print "Next";
Print substr($ip, 0, 7);
?>

After
