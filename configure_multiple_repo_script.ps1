
Write-Output "Remote repositories before setup."
git remote -v

$confirmation = Read-Host "Are you Sure You Want To Proceed: (y/n)"

if ($confirmation -eq 'y') {
    
	$primary_repo=Read-Host -Prompt "Enter primary repository url. This is the actual repository holding the source code.";
	$secondary_repo=Read-Host -Prompt "Enter secondary repository url. This is the secondary repository where the changes needs to be synchronized.";



	git remote set-url --add --push origin $primary_repo
	git remote set-url --add --push origin $secondary_repo

	Write-Output "Remote repositories after setup."
	git remote -v
}